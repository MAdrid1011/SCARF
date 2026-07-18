#!/usr/bin/env python3
"""
SCARF Demo - Hardware Simulator for 3D Gaussian Splatting Encoders

SCARF (Scene-Adaptive Cost-volume Accelerator with Reuse Framework) is a
hardware-realizable accelerator for 3D Gaussian Splatting encoders.

This demo shows SCARF's two key optimizations:
1. SAES (Scene-Adaptive Early Sparsification): Skip depth search for homogeneous tiles
2. FSDR (Feature-Similarity Depth Reuse): Cache and reuse depth for similar pixels

Data Flow:
    [Neural Network Backbone] → features
           ↓
    [SCARF SAES] → tile decisions (early-stop vs continue)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Early-Stop     Continue
    ↓             ↓
Skip S2     [S2 Depth + FSDR] → depths
    ↓             ↓
[SCARF GGU]  [SCARF GGU]
(representative) (full)
    ↓             ↓
    └──────┬──────┘
           ↓
    Gaussians + Cycle Counts
           ↓
    [Decoder] → Rendered Image

Output Metrics:
- Quality: PSNR, SSIM (vs baseline without SCARF)
- Performance: Hardware cycle counts (S1, S2, S3, GGU, total)
- SAES: Early-stop ratio, Gaussian reduction
- FSDR: Cache hit rate, Memory access reduction

Usage:
    python demo.py [--model MODEL_TYPE] [OPTIONS]

Supported Models:
    - transplat (default)
    - mvsplat
    - depthsplat

Fallback Options (for debugging):
    --no-feature      Disable SCARF feature extraction HW simulator, use original GPU
    --no-depth        Disable SCARF depth prediction HW simulator, use original GPU
    --no-gaussian     Disable SCARF gaussian generation HW simulator (GGU), use original GPU
    --no-saes         Disable SAES early sparsification
    --no-fsdr         Disable FSDR depth reuse
    --baseline-only   Only run baseline, skip SCARF pipeline
"""

import sys
import os
from pathlib import Path

# SCARF root directory
SCARF_ROOT = Path(__file__).parent.parent.resolve()

# All network-fetched model assets are prepared and hash-checked in advance.
os.environ.setdefault('TORCH_HOME', str(SCARF_ROOT / 'assets' / 'torch'))

# Add SCARF to path for imports
sys.path.insert(0, str(SCARF_ROOT))

from scripts.demo_cli import parse_args as parse_demo_args

# Keep the public CLI inspectable before heavyweight model dependencies exist.
if any(option in sys.argv[1:] for option in ("-h", "--help")):
    parse_demo_args()

import torch
import torch.nn.functional as F
from einops import rearrange
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import argparse

# SCARF imports
from adapters import create_adapter, BaseAdapter
from integration import (
    create_model_loader,
    load_target_free_calibration_data,
    ModelBundle,
    DataBundle,
)
from ggu import GGUProcessor, GGUConfig
from fsdr import FSDRSimulator
from saes import ProgressiveSAES, apply_progressive_saes
from saes.execution_dependency import (
    resolve_s2_s3_execution_contract,
    s2_s3_saving_ratio,
)
from saes.hardware_accounting import build_saes_event_ledger
from scripts.reproducibility import capture_torch_rng_state, restore_torch_rng_state
from scripts.result_record import strict_stage_error

# SCARF hardware clock frequency (MHz) — default 1 GHz, overridable via --freq
SCARF_FREQ_MHZ = 1000


def is_target_rgb_free_execution(args: argparse.Namespace) -> bool:
    """Return whether the requested path must remove target RGB before execution."""
    return bool(
        args.fsdr_only
        or args.saes_materialization_audit
        or args.saes_s3_raw_audit
        or args.saes_routing_audit
        or args.saes_hardware_audit
    )


# ============================================================
# GPU Timing & Info Utilities
# ============================================================
def get_gpu_info() -> Dict[str, object]:
    """
    Query GPU name and max SM clock frequency.
    
    Returns dict with keys: 'name' (str), 'freq_mhz' (int).
    Falls back to defaults if nvidia-smi is unavailable.
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.max.sm,gpu_name',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            freq_mhz = int(parts[0].strip().replace(' MHz', ''))
            name = parts[1].strip()
            return {'name': name, 'freq_mhz': freq_mhz}
    except Exception:
        pass

    if torch.cuda.is_available():
        for path in Path("/sys/devices").glob("**/devfreq/*gpu*/max_freq"):
            try:
                raw = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            freq_mhz = raw / 1_000_000 if raw >= 1_000_000 else raw / 1_000
            if freq_mhz > 0:
                return {'name': torch.cuda.get_device_name(0), 'freq_mhz': freq_mhz}
        raise RuntimeError("cannot determine GPU clock from nvidia-smi or device sysfs")
    return {'name': 'CPU', 'freq_mhz': 0}


def gpu_timed_inference(
    fn,
    warmup: int = 1,
    repeats: int = 5,
    device: Optional[torch.device] = None,
    return_samples: bool = False,
):
    """
    Precisely measure GPU inference time using CUDA events.
    
    Args:
        fn: Callable (no args) that runs the inference under torch.no_grad().
        warmup: Number of warmup iterations (not timed).
        repeats: Number of timed iterations; returns the median.
    
    Returns:
        Median GPU time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
    if not use_cuda:
        import statistics

        times_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.no_grad():
                fn()
            times_ms.append((time.perf_counter() - start) * 1000.0)
        median = statistics.median(times_ms)
        return (median, times_ms) if return_samples else median

    torch.cuda.synchronize()

    # Timed runs
    times_ms: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    # Return median
    import statistics

    median = statistics.median(times_ms)
    return (median, times_ms) if return_samples else median


# ============================================================
# Configuration
# ============================================================
@dataclass
class SCARFConfig:
    """SCARF pipeline configuration."""
    # Model type
    model_type: str = 'transplat'
    
    # ---- Hardware Performance Config (High-Performance ASIC) ----
    # Upgraded compute arrays: 48×48 MAC (ConvEngine + GEMM = 4608 MACs total)
    # 64-wide Vector ALU, 64ch×32 BilinearUnit, 32 GGU PEs
    #
    # Differentiated scaling vs base 32×32 config:
    #   Compute-bound stages (conv, GEMM, depth_head, regression, GGU):
    #     48²/32² = 2.25x MACs → 2.0x effective (dataflow overhead)
    #   Memory-bound stages (cost_volume bilinear warping, feature reads):
    #     SRAM bandwidth scales ~1.5x (wider buses: 48/32=1.5x input row width)
    #     Memory-bound speedup limited by SRAM port throughput
    hw_scale_compute: float = 2.0   # Compute-bound: conv, GEMM, depth_head, etc.
    hw_scale_memory: float = 1.5    # Memory-bound: cost_volume, bilinear sampling
    ggu_pe_count: int = 32
    
    # ---- S1 CNN-Specific ASIC Optimizations ----
    # Additional acceleration for S1 Feature Extraction on ASIC, exploiting
    # regular dataflow patterns that GPUs cannot fully leverage.
    #
    # CNN Boost (1.8x additional on top of HW_SCALE_C):
    #   1. Winograd F(2,3) for 3×3 convolutions:
    #      - 2.25x fewer multiplications (4 mults instead of 9 per output)
    #      - Pre/post transforms hardwired as combinational adder trees (negligible area)
    #      - Net ~1.5x after transform overhead and non-3×3 layers (Conv7x7 initial)
    #      - Refs: Lavin & Gray CVPR'16; Lu et al. TCAS-I'18
    #   2. Conv-BN-ReLU layer fusion:
    #      - BN scale/shift + ReLU pipelined into systolic array output stage
    #      - Eliminates intermediate SRAM write/read between conv, norm, activation
    #      - ~1.12x from removing norm+activation buffer traffic
    #   3. Weight-stationary dataflow:
    #      - Weights loaded once to PE array, input feature maps stream through
    #      - Amortizes weight load cost across all spatial positions
    #      - ~1.05x from reduced weight reload overhead
    #   Combined: 1.5 × 1.12 × 1.05 ≈ 1.76 → rounded to 1.8x
    #
    # ViT/Transformer Boost (1.15x additional):
    #   1. Large regular GEMM on systolic array: 1.1x
    #      - Attention Q/K/V projections and MLP are large dense GEMMs
    #      - Near-peak utilization (>90%) vs CNN's varied kernel shapes
    #   2. Fused LayerNorm + GELU activation: 1.05x
    #      - Similar to Conv-BN-ReLU fusion but for transformer blocks
    #   Combined: 1.1 × 1.05 ≈ 1.15x
    s1_cnn_boost: float = 1.8    # CNN: Winograd + fusion + weight-stationary
    s1_vit_boost: float = 1.15   # ViT/Transformer: GEMM efficiency + fusion
    
    # ---- SAES v4 multi-level config (L0+L1 only) ----
    tile_size: int = 4
    saes_cov_safety: float = 1.02     # Safety factor for interpolated covariances
    feature_var_threshold: float = 0.20   # Level 0: natural saturation (~18-34% tiles, model-dependent)
    depth_std_threshold: float = 0.10     # Paper default tau_d
    saes_cross_check: float = 0.015       # Probe cross-check error threshold (conservative, ≤2% quality across all models)
    
    # ---- FSDR config — Depth-Only Reuse (Realistic ASIC) ----
    # In real ASIC: cache hit → reuse cached DEPTH (skip S2 only)
    # S3 still runs with cached depth → only means/positions affected
    # Quality impact is proportional to depth error (very small for matched pixels)
    # This allows much more aggressive reuse criteria than full-Gaussian reuse
    fsdr_cache_size: int = 32
    fsdr_hamming_threshold: int = 3       # Paper default tau_h
    fsdr_reuse_hamming: int = 3           # Moderate hamming (depth-only is safe)
    fsdr_reuse_spatial: int = 12          # Moderate spatial distance
    fsdr_reuse_confidence: float = 0.80   # Min peak_prob for reuse decision
    fsdr_seed: int = 42                   # Public FP16 hyperplane ROM seed
    gamma_depth: float = 0.10             # Local-validity tolerance selected by calibration
    beta_x: float = 0.50                  # Tile-normalized spatial assignment bandwidth
    beta_f: float = 0.10                  # Normalized feature assignment bandwidth
    beta_d: float = 1.00                  # L1 depth-reliability multiplier
    # All FSDR-guided pixels use single-tier narrowed depth search (no S3 bypass).
    # fsdr_tier1_ratio removed: Tier 1 (HD ≤ 1 full Gaussian reuse) is not
    # implemented in the simulation; claiming it would overstate speedup.
    
    # Depth prediction config (will be overridden based on model type)
    num_depth_candidates: int = 32  # Default for MVSplat; TranSplat/DepthSplat use 128
    fsdr_narrowed_candidates: int = 8  # FSDR narrowed search target (D/4)
    feature_dim: int = 128
    
    # GGU config (defaults, overridden by adapter)
    scale_min: float = 0.5
    scale_max: float = 15.0
    sh_degree: int = 4
    
    # Test config
    
    def apply_adapter_overrides(self, adapter: BaseAdapter):
        """Apply model-specific configuration from adapter."""
        fsdr_overrides = adapter.get_fsdr_config_overrides()
        saes_overrides = adapter.get_saes_config_overrides()
        scale_min, scale_max = adapter.get_scale_range()
        
        # Apply overrides
        if 'hamming_threshold' in fsdr_overrides:
            self.fsdr_hamming_threshold = fsdr_overrides['hamming_threshold']
        if 'reuse_hamming' in fsdr_overrides:
            self.fsdr_reuse_hamming = fsdr_overrides['reuse_hamming']
        if 'reuse_spatial' in fsdr_overrides:
            self.fsdr_reuse_spatial = fsdr_overrides['reuse_spatial']
        if 'early_stop_threshold' in saes_overrides:
            pass  # L2 Gaussian-similarity threshold removed; saes_overrides ignored
        
        self.scale_min = scale_min
        self.scale_max = scale_max


# Global config (initialized in main)
CONFIG: SCARFConfig = None
_MODEL_BUNDLE_CACHE = {}
_LPIPS_METRIC_CACHE = {}


def get_lpips_metric(device: torch.device):
    key = str(device)
    metric = _LPIPS_METRIC_CACHE.get(key)
    if metric is None:
        from lpips import LPIPS

        metric = LPIPS(net="vgg").to(device).eval()
        _LPIPS_METRIC_CACHE[key] = metric
    return metric


# ============================================================
# GGU Cycle Counter (Hardware Performance)
# ============================================================
class HWCycleCounter:
    """
    Hardware cycle counter for GGU (Gaussian Generation Unit).
    
    Models parallelism via ``ggu_pe_count`` parallel processing elements,
    aligned with ConvEngine (16x16 PE) and GEMMUnit (8x16 tile).
    
    Note: Depth prediction cycles come from HWDepthPredictor directly;
    S2 depth prediction cycles come from the depth predictor's
    cost_volume + regression cycle estimates.
    """
    
    # Per-Gaussian cycle budget for dedicated GGU Processing Elements:
    #   Each PE has a small 3x3 matmul unit (not the 32x32 systolic array).
    #   Per Gaussian the PE executes:
    #     - position:    ray origin + direction * depth (3 FMA + normalize)  = ~10 cycles
    #     - scale:       sigmoid LUT + depth scaling                        = ~5 cycles
    #     - quat2mat:    quaternion normalize + 9 entries (50 mul, 20 add)  = ~30 cycles
    #     - covariance:  R @ S @ S^T @ R^T  (3 x 3×3 matmul @ 12c each)  = ~36 cycles
    #     - transform:   R_c2w @ cov @ R_c2w^T  (2 x 3×3 matmul)          = ~24 cycles
    #     - sh_rotation: rotate 25 SH coeffs × 3 channels (band-wise)      = ~80 cycles
    #     - opacity:     sigmoid LUT                                        = ~2 cycles
    #   TOTAL per Gaussian ≈ 187 cycles
    GGU_CYCLES = {
        'position': 10,     # Ray casting (3 FMA + normalize)
        'scale': 5,         # Sigmoid LUT + depth scaling
        'quat2mat': 30,     # Quaternion → rotation matrix (50 mul + 20 add)
        'covariance': 36,   # R @ S @ S^T @ R^T (3 × 3×3 matmul)
        'transform': 24,    # R_c2w @ cov @ R_c2w^T (2 × 3×3 matmul)
        'sh_rotation': 80,  # Rotate 25 SH coeffs × 3 channels
        'opacity': 2,       # Sigmoid LUT
    }
    
    GGU_PIPELINE_OVERHEAD = 4   # cycles to fill/drain the GGU pipeline per batch
    
    def __init__(self, ggu_pe_count: int = 32):
        self.ggu_pe_count = ggu_pe_count
        self.reset()
    
    def reset(self):
        self._ggu_raw_cycles = 0
        self._ggu_elements = 0
        self._compact_bytes_saved  = 0
        self._compact_cycles_saved = 0
    
    def add_ggu(self, num_gaussians: int = 1, sh_degree: int = 4):
        """
        Record cycles for GGU gaussian generation.

        Also computes compact-writeback savings (Dataflow spec §Stage 4):
          - Covariance: symmetric 3×3 → only 6 upper-triangle elements written
            (saves 3 floats × 4 B = 12 B per Gaussian vs. full 9-element write).
          - SH: only (sh_degree+1)² × 3 coefficients written (effective terms).
            Saves (25 − (sh_degree+1)²) × 3 × 4 B per Gaussian for degree < 4.
          - Writeback cycles modelled as 1 cycle per 4-byte float saved.
        """
        cycles_per_gaussian = sum(self.GGU_CYCLES.values())
        total_cycles = cycles_per_gaussian * num_gaussians
        self._ggu_raw_cycles += total_cycles
        self._ggu_elements += num_gaussians

        # Compact writeback bandwidth savings
        cov_floats_saved  = 3                                    # 9→6 upper-triangle
        sh_full_coeffs    = 25 * 3                               # degree-4 full SH (3 channels)
        sh_eff_coeffs     = (sh_degree + 1) ** 2 * 3            # effective coefficients
        sh_floats_saved   = max(0, sh_full_coeffs - sh_eff_coeffs)
        floats_saved_per_gaussian = cov_floats_saved + sh_floats_saved
        compact_bytes_saved = floats_saved_per_gaussian * 4 * num_gaussians  # 4 B per float
        compact_cycles_saved = floats_saved_per_gaussian * num_gaussians     # 1 cy per float

        # Accumulate compact savings (reported in get_summary)
        self._compact_bytes_saved  = getattr(self, '_compact_bytes_saved',  0) + compact_bytes_saved
        self._compact_cycles_saved = getattr(self, '_compact_cycles_saved', 0) + compact_cycles_saved

        return total_cycles
    
    def _parallel_cycles(self, raw_cycles: int, num_elements: int,
                         pe_count: int, pipeline_overhead: int) -> int:
        """Convert raw serial cycles to parallel hardware cycles."""
        if num_elements == 0 or raw_cycles == 0:
            return 0
        compute = (raw_cycles + pe_count - 1) // pe_count
        num_batches = (num_elements + pe_count - 1) // pe_count
        overhead = num_batches * pipeline_overhead
        return compute + overhead
    
    def get_ggu_cycles(self) -> int:
        """Get parallel-adjusted GGU cycles for ALL Gaussians."""
        return self._parallel_cycles(
            self._ggu_raw_cycles, self._ggu_elements,
            self.ggu_pe_count, self.GGU_PIPELINE_OVERHEAD,
        )
    
    def get_summary(self) -> Dict:
        return {
            'ggu_cycles': self.get_ggu_cycles(),
            'ggu_raw_cycles': self._ggu_raw_cycles,
            'ggu_pe_count': self.ggu_pe_count,
            'ggu_elements': self._ggu_elements,
            'compact_writeback_bytes_saved':  getattr(self, '_compact_bytes_saved',  0),
            'compact_writeback_cycles_saved': getattr(self, '_compact_cycles_saved', 0),
        }


# ============================================================
# Savings Tracker (SAES + FSDR cycle savings model)
# ============================================================
class SavingsTracker:
    """
    Track cycle savings from SAES v4 (multi-level, dataflow-aligned) and FSDR optimizations.

    v4 changes (vs v3):
    - K(T) adaptive probe count: K(T) = 4 + ceil(2·log2(T/4))
    - L0/L1: weighted moment matching (space+feature / space+feature+depth weights)
             + covariance spread term (law of total variance) for coverage.
    - L0 retains K(T) anchors; the less-compressive L1 path retains 2K(T).
    - FSDR: Hamming hits narrow S2 to D/4 candidates on the paper/RTL path.
    """
    
    LIGHT_VERIFY_COST_RATIO = 0.04
    
    def __init__(self, *, model_type: str | None = None):
        self.total_pixels = 0
        # SAES v4: L0 + L1 only
        self.level0_pixels = 0   # Feature-uniform: K(T) anchors, 12/16 removed at T=4.
        self.level1_pixels = 0   # Depth-uniform: 2K(T) anchors, 8/16 removed at T=4.
        self.saes_interpolated_pixels = 0  # Total modified (backward compat)
        # FSDR
        self.fsdr_direct_reuse = 0
        self.fsdr_interpolation = 0
        self.fsdr_light_verify = 0
        self.fsdr_full_search = 0
        self.fsdr_validated = 0
        self.fsdr_rejected = 0
        # Populated only from executed SAES tile counters.  This is an
        # analytic no-overlap ledger, not RTL-cycle-equivalent timing.
        self.saes_hardware_ledger = None
        self.saes_execution_dependency = resolve_s2_s3_execution_contract(
            model_type
        )
    
    def record_saes(
        self,
        total_pixels: int,
        saes_stats: Dict,
        *,
        feature_dim: int | None = None,
        tile_size: int | None = None,
        sh_degree: int | None = None,
        primitives_per_pixel: int = 1,
        model_type: str | None = None,
    ) -> Dict | None:
        """Record SAES v4 (L0+L1) statistics."""
        if model_type is not None:
            requested_contract = resolve_s2_s3_execution_contract(model_type)
            current_model = self.saes_execution_dependency["model"]
            if current_model not in (None, requested_contract["model"]):
                raise ValueError(
                    "SAES execution-dependency model changed within one tracker"
                )
            self.saes_execution_dependency = requested_contract
        self.total_pixels = total_pixels
        validated = saes_stats.get('validated_modified_pixels', None)
        if validated is not None:
            raw_total = (saes_stats.get('level0_pixels', 0) +
                         saes_stats.get('level1_pixels', 0))
            ratio = validated / raw_total if raw_total > 0 else 0.0
            self.level0_pixels = int(saes_stats.get('level0_pixels', 0) * ratio)
            self.level1_pixels = int(saes_stats.get('level1_pixels', 0) * ratio)
        else:
            self.level0_pixels = saes_stats.get('level0_pixels', 0)
            self.level1_pixels = saes_stats.get('level1_pixels', 0)
        self.saes_interpolated_pixels = self.level0_pixels + self.level1_pixels
        dimensions = (feature_dim, tile_size, sh_degree)
        if any(value is not None for value in dimensions):
            if any(value is None for value in dimensions):
                raise ValueError(
                    "SAES accounting requires feature_dim, tile_size, and sh_degree together"
                )
            self.saes_hardware_ledger = build_saes_event_ledger(
                saes_stats,
                feature_dim=int(feature_dim),
                tile_size=int(tile_size),
                sh_degree=int(sh_degree),
                primitives_per_pixel=int(primitives_per_pixel),
            )
        else:
            self.saes_hardware_ledger = None
        return self.saes_hardware_ledger
    
    def record_fsdr_pixel(self, path: str, reused: bool = False, validated: bool = False):
        """Record one pixel's FSDR path with reuse status.
        
        Args:
            path: Decision path ('reuse', 'hit_no_reuse', 'full_compute', etc.)
            reused: True if this pixel used cached Gaussians (skipped S2+S3)
            validated: Deprecated alias for reused (backward compat)
        """
        if path in ('reuse', 'guided'):
            self.fsdr_direct_reuse += 1
        elif path in ('hit_no_reuse', 'hit_no_guide', 'full_compute', 'full_search'):
            self.fsdr_full_search += 1
        else:
            self.fsdr_full_search += 1
        if reused or validated:
            self.fsdr_validated += 1
    
    @property
    def fsdr_total(self) -> int:
        """Total pixels processed by FSDR."""
        return (self.fsdr_direct_reuse + self.fsdr_interpolation +
                self.fsdr_light_verify + self.fsdr_full_search)
    
    # --- Per-level saving ratios ---
    def level0_ratio(self) -> float:
        """Fraction of pixels replicated by Level 0 (feature-uniform)."""
        return self.level0_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def level1_ratio(self) -> float:
        """Fraction of pixels interpolated by Level 1 (depth-uniform)."""
        return self.level1_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def saes_interpolation_ratio(self) -> float:
        """Total fraction of pixels modified by SAES (all levels)."""
        return self.saes_interpolated_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def fsdr_validated_saving_ratio(self) -> float:
        """Fraction of ALL pixels validated as fully skippable by FSDR (S2+S3)."""
        return self.fsdr_validated / self.total_pixels if self.total_pixels > 0 else 0.0
    
    
    def compute_ablation(self, feature_cycles: int, depth_cycles: int,
                         ggu_cycles: int,
                         dp_core_cycles: int = 0,
                         gauss_gen_cycles: int = 0,
                         cost_volume_cycles: int = 0,
                         s1_cnn_cycles: int = 0,
                         ablation_quality: Dict = None,
                         fsdr_narrowing_ratio: float = 0.75) -> Dict[str, Dict]:
        """
        Compute cycle counts for all ablation configurations.
        
        High-Performance ASIC Pipeline (upgraded hardware):
          - 48×48 ConvEngine + 48×48 GEMM = 4608 MACs total
          - 64-wide Vector ALU, 64ch×32 BilinearUnit, 32 GGU PEs
          - Differentiated HW scaling: compute-bound 2.0x, memory-bound 1.5x
        
        3-Stage Pipeline:
          Stage 1: Feature Extraction (FE) - ConvEngine + GEMM
          Stage 2: Depth Prediction (cost_volume + unet + depth_head + regression)
          Stage 3: Gaussian Generation (refine_unet + to_gaussians + GGU post-processing)
        
        SAES accounting model:
          SAES Level 0 and Level 1 route counters describe K(T)/2K(T)
          retained-anchor work.  They reduce S2/S3 cycles only if the selected
          model's target-free execution-dependency contract proves that the
          current sparse path does not consume skipped positions.  Otherwise
          their S2/S3 saving is fail-closed to zero while their control, merge,
          and storage ledger remains charged.
          FSDR (Narrowed Search): Guided pixels search D/4 depth candidates.
                Saves cost_volume computation only (memory-bound, per-pixel per-candidate).
                U-Net/depth_head/regression unchanged (process full spatial resolution).
                Decision: LSH Hamming on the paper/RTL path. The historical
                depth-consistency guard is isolated from claim execution.
        """
        # ================================================================
        # Differentiated hardware compute scaling (48×48 upgraded arrays)
        # ================================================================
        # Rationale for separate compute vs memory scaling:
        #   48×48 systolic array has 2.25x MACs vs 32×32 base.
        #   Compute-bound stages (dominated by MAC operations) scale ~2.0x
        #   (2.25x MACs - ~11% dataflow/utilization overhead).
        #   Memory-bound stages (dominated by SRAM reads/bilinear warping)
        #   scale ~1.5x (SRAM input bandwidth ∝ array row width: 48/32=1.5x).
        HW_SCALE_C = CONFIG.hw_scale_compute  # 2.0x for compute-bound
        HW_SCALE_M = CONFIG.hw_scale_memory   # 1.5x for memory-bound
        
        # Route coverage is not an executable bypass claim.  The model-specific
        # dependency contract below decides whether any of it can reduce S2/S3.
        l0_ratio = self.level0_ratio()
        l1_ratio = self.level1_ratio()
        total_saes = l0_ratio + l1_ratio
        if total_saes > 0.0 and self.saes_hardware_ledger is None:
            raise ValueError(
                "nonzero SAES savings require an explicit control/storage event ledger"
            )
        saes_accounting_cycles = (
            int(
                self.saes_hardware_ledger['cycles'][
                    'serialized_accounting_cycles'
                ]
            )
            if self.saes_hardware_ledger is not None
            else 0
        )
        
        # FSDR can exclude only positions that an executable SAES S2 bypass
        # actually removed.  Route coverage without a verified sparse path
        # remains in the FSDR denominator.
        fsdr_ratio = self.fsdr_validated_saving_ratio()
        
        if min(feature_cycles, dp_core_cycles, gauss_gen_cycles, ggu_cycles) <= 0:
            raise ValueError("complete positive S1, S2, S3, and GGU cycles are required")
        if not 0 < s1_cnn_cycles <= feature_cycles:
            raise ValueError("S1 CNN cycle breakdown is missing or inconsistent")
        if not 0 < cost_volume_cycles < dp_core_cycles:
            raise ValueError("S2 cost-volume cycle breakdown is missing or inconsistent")
        
        # ================================================================
        # Apply differentiated hardware scaling per stage
        # ================================================================
        # S1 (Feature Extraction): CNN-specific ASIC optimizations
        #   CNN portion: Winograd + Conv-BN-ReLU fusion + weight-stationary
        #   ViT portion: Large GEMM efficiency + LayerNorm+GELU fusion
        S1_CNN_BOOST = CONFIG.s1_cnn_boost   # 1.8x additional for CNN
        S1_VIT_BOOST = CONFIG.s1_vit_boost   # 1.15x additional for ViT
        
        s1_vit_cycles = feature_cycles - s1_cnn_cycles
        fe_cnn = int(s1_cnn_cycles / (HW_SCALE_C * S1_CNN_BOOST))
        fe_vit = int(s1_vit_cycles / (HW_SCALE_C * S1_VIT_BOOST))
        feature_cycles = fe_cnn + fe_vit
        
        # S2 (Depth Prediction): split cost_volume (memory-bound) from rest (compute-bound)
        #   cost_volume: bilinear warping + correlation → limited by SRAM read bandwidth
        #   unet + depth_head + regression: convolutions/GEMM → compute-bound
        non_cv_cycles = dp_core_cycles - cost_volume_cycles
        dp_cv_scaled = int(cost_volume_cycles / HW_SCALE_M)   # memory-bound
        dp_rest_scaled = int(non_cv_cycles / HW_SCALE_C)      # compute-bound
        dp_core_cycles = dp_cv_scaled + dp_rest_scaled
        
        # S3 (Gaussian Generation): refine_unet + to_gaussians → compute-bound
        gauss_gen_cycles = int(gauss_gen_cycles / HW_SCALE_C)
        
        # GGU post-processing: dedicated PEs → compute-bound
        ggu_cycles = int(ggu_cycles / HW_SCALE_C)
        
        # ================================================================
        # Pipeline overlap factors
        # ================================================================
        # Architectural basis for each factor:
        #
        # PIPE_FE = 0.95: S1 has CNN (ConvEngine) + ViT (GEMM Unit) sub-stages.
        #   These use different HW units and can overlap at tile boundaries via
        #   dual-port SRAM double-buffering. 5% overlap is conservative for
        #   inter-unit pipelining within a single stage.
        #
        # PIPE_DP = 0.88: S2 has cost_volume (BilinearUnit) → UNet (ConvEngine) →
        #   depth_head (ConvEngine) → regression (GEMM). Cost_volume for tile N+1
        #   can overlap with UNet processing of tile N via ping-pong buffers.
        #   12% overlap from tile-level pipelining across 4 sequential sub-stages.
        #
        # PIPE_GG_NN = 0.97: S3 is mostly sequential (refine_unet → to_gaussians).
        #   3% overlap from output buffer write-back overlapping with next-tile
        #   weight prefetch. Minimal because both sub-stages share ConvEngine.
        #
        # Note: these are architectural estimates, not cycle-accurate simulation.
        PIPE_FE = 0.95
        PIPE_DP = 0.88
        PIPE_GG_NN = 0.97
        
        # ================================================================
        # Per-level cycle savings computation
        # ================================================================
        # The pixel ratios represent the fraction of SKIPPED non-probe pixels:
        #   l0_ratio = (num_L0_tiles * non_probe_count) / total_pixels
        #   l1_ratio = (num_L1_tiles * non_probe_count) / total_pixels
        #
        # A SAES route is not a direct S2/S3 bypass unless the current model
        # has independently demonstrated that retained outputs do not depend
        # on skipped positions.  Current contracts intentionally fail closed.
        saes_s2_saving = s2_s3_saving_ratio(
            total_saes, self.saes_execution_dependency
        )
        saes_s3_saving = s2_s3_saving_ratio(
            total_saes, self.saes_execution_dependency
        )
        
        # ================================================================
        # FSDR savings: Narrowed Depth Search only
        # ================================================================
        # The FSDR simulation implements a single-tier narrowed search:
        #   All guided pixels run S2 with D/4 candidates (75% S2 savings).
        #   S3 (Gaussian generation) still runs for every pixel — there is no
        #   Tier-1 S3 bypass in the current simulation.  Claiming S3 savings
        #   for any fraction of guided pixels would overstate the speedup.
        #
        # Architecture: LSH hash unit (16-bit) + 32-entry default cache table.
        #   Guided (cache hit, hamming ≤ reuse_hamming): narrowed D/4 search.
        #   Miss: full D-candidate search.
        #
        # Per-pixel S2 saving for guided pixels (D-dependent ops only):
        #   depth_dep_frac ≈ cv_frac + 0.04  (cost-volume + depth head + regression)
        #   tier_saving    = depth_dep_frac × fsdr_narrowing_ratio
        # S3 saving: 0 (not implemented; narrowed search still produces depth
        #   that feeds S3 normally).
        FSDR_TIER2_RATIO = 1.0  # All guided pixels use narrowed search

        cv_frac_scaled = dp_cv_scaled / dp_core_cycles

        # depth_dep_frac: fraction of S2 that depends on D (depth candidates).
        # Components:
        #   cv_frac_scaled : cost-volume correlation (explicit D-dependent memory op)
        #   + 0.75         : depth UNet (processes 3D cost volume H×W×D → D-dependent)
        #                    + depth head (softmax over D planes)
        #                    + regression (weighted sum over D candidates)
        # For models where UNet is not D-dependent, this is a slight over-estimate;
        # for all DepthSplat/MVSplat/TranSplat architectures the UNet takes the
        # D-plane cost volume as input, so D-dependent modeling is accurate.
        depth_dep_frac = min(cv_frac_scaled + 0.75, 1.0)
        tier2_per_pixel = depth_dep_frac * fsdr_narrowing_ratio

        # S2 saving: all guided pixels get D/4 narrowed search on ALL D-dependent ops
        # S3 saving: 0 — simulation does not skip S3 for any FSDR-guided pixel
        FSDR_S2_SAVE_PER_PIXEL = FSDR_TIER2_RATIO * tier2_per_pixel
        FSDR_S3_SAVE_PER_PIXEL = 0.0

        remaining_for_fsdr = 1.0 - saes_s2_saving
        fsdr_s2_saving = fsdr_ratio * remaining_for_fsdr * FSDR_S2_SAVE_PER_PIXEL
        fsdr_s3_saving = fsdr_ratio * remaining_for_fsdr * FSDR_S3_SAVE_PER_PIXEL
        
        # ================================================================
        # Build ablation configs
        # ================================================================
        configs = {}
        
        def _make_cfg(
            fe,
            dp,
            gg_nn,
            ggu_post,
            d_sav,
            g_sav,
            cfg_key=None,
            saes_control=0,
        ):
            """Build config with both raw (serial) and effective (pipelined) cycles."""
            if saes_control < 0:
                raise ValueError("SAES control accounting cycles must be nonnegative")
            raw_total = fe + dp + gg_nn + ggu_post + saes_control
            
            eff_fe = int(fe * PIPE_FE)
            eff_dp = int(dp * PIPE_DP)
            eff_gg_nn = int(gg_nn * PIPE_GG_NN)
            eff_ggu = 0  # hidden behind conv work
            # The ledger deliberately assumes no overlap with S2/S3/VectorALU
            # work.  It makes the analytic SAES ablation stricter, but does not
            # turn the number into an RTL timing claim.
            eff_total = eff_fe + eff_dp + eff_gg_nn + eff_ggu + saes_control
            
            result = {
                'feature': fe,
                'dp_core': dp,
                'gauss_gen': gg_nn + ggu_post,
                'total': raw_total,
                'depth_saving': d_sav,
                'gauss_saving': g_sav,
                'gauss_head_nn': gg_nn,
                'ggu_post': ggu_post,
                'saes_control': saes_control,
                'eff_feature': eff_fe,
                'eff_dp_core': eff_dp,
                'eff_gauss_gen': eff_gg_nn,
                'eff_total': eff_total,
                'pipeline_saving': 1.0 - eff_total / raw_total if raw_total > 0 else 0.0,
                'hw_scale_compute': HW_SCALE_C,
                'hw_scale_memory': HW_SCALE_M,
            }
            
            if ablation_quality and cfg_key and cfg_key in ablation_quality:
                result['quality'] = ablation_quality[cfg_key]
            
            return result
        
        # 1. ASIC (no optimizations — but with HW-scaled cycles)
        configs['asic'] = _make_cfg(
            feature_cycles, dp_core_cycles, gauss_gen_cycles, ggu_cycles,
            0.0, 0.0, cfg_key='asic')
        
        # 2. ASIC + FSDR only (single-tier narrowed depth search)
        fsdr_alone_s2 = fsdr_ratio * FSDR_S2_SAVE_PER_PIXEL
        fsdr_alone_s3 = fsdr_ratio * FSDR_S3_SAVE_PER_PIXEL
        dp_fsdr = int(dp_core_cycles * (1.0 - fsdr_alone_s2))
        gh_fsdr = int(gauss_gen_cycles * (1.0 - fsdr_alone_s3))
        ggu_fsdr = int(ggu_cycles * (1.0 - fsdr_alone_s3))
        configs['asic_fsdr'] = _make_cfg(
            feature_cycles, dp_fsdr, gh_fsdr, ggu_fsdr,
            fsdr_alone_s2, fsdr_alone_s3, cfg_key='asic_fsdr')
        
        # 3. ASIC + SAES only (multi-level)
        dp_saes = int(dp_core_cycles * (1.0 - saes_s2_saving))
        gh_saes = int(gauss_gen_cycles * (1.0 - saes_s3_saving))
        ggu_saes = int(ggu_cycles * (1.0 - saes_s3_saving))
        configs['asic_saes'] = _make_cfg(
            feature_cycles, dp_saes, gh_saes, ggu_saes,
            saes_s2_saving, saes_s3_saving, cfg_key='asic_saes',
            saes_control=saes_accounting_cycles)
        
        # 4. ASIC + SAES + FSDR (full optimization)
        combined_s2_saving = saes_s2_saving + fsdr_s2_saving
        combined_s3_saving = saes_s3_saving + fsdr_s3_saving  # fsdr_s3_saving=0
        dp_both = int(dp_core_cycles * (1.0 - combined_s2_saving))
        dp_both = max(0, dp_both)
        gh_both = int(gauss_gen_cycles * (1.0 - combined_s3_saving))
        gh_both = max(0, gh_both)
        ggu_both = int(ggu_cycles * (1.0 - combined_s3_saving))
        ggu_both = max(0, ggu_both)
        configs['asic_fsdr_saes'] = _make_cfg(
            feature_cycles, dp_both, gh_both, ggu_both,
            combined_s2_saving, combined_s3_saving,
            cfg_key='asic_fsdr_saes', saes_control=saes_accounting_cycles)
        
        # Store pipeline factors and multi-level info
        configs['_pipeline'] = {
            'PIPE_FE': PIPE_FE,
            'PIPE_DP': PIPE_DP,
            'PIPE_GG_NN': PIPE_GG_NN,
            'GGU_hidden': True,
            'HW_SCALE_C': HW_SCALE_C,
            'HW_SCALE_M': HW_SCALE_M,
            'S1_CNN_BOOST': S1_CNN_BOOST,
            'S1_VIT_BOOST': S1_VIT_BOOST,
            'cv_frac_scaled': cv_frac_scaled,
            'fsdr_s2_save_per_pixel': FSDR_S2_SAVE_PER_PIXEL,
            'saes_l0_ratio': l0_ratio,
            'saes_l1_ratio': l1_ratio,
            'saes_total_ratio': total_saes,
            'saes_s2_saving': saes_s2_saving,
            'saes_s3_saving': saes_s3_saving,
            'saes_execution_dependency': self.saes_execution_dependency,
            'saes_control_accounting_cycles': saes_accounting_cycles,
            'saes_control_timing_class': (
                self.saes_hardware_ledger['timing_class']
                if self.saes_hardware_ledger is not None
                else 'not_applicable_no_saes_savings'
            ),
            'fsdr_reuse_ratio': fsdr_ratio,
            'fsdr_s2_saving': fsdr_s2_saving,
            'fsdr_s3_saving': fsdr_s3_saving,
            'combined_s2_saving': combined_s2_saving,
            'combined_s3_saving': combined_s3_saving,
        }
        configs['_saes_accounting'] = self.saes_hardware_ledger
        
        return configs


# ============================================================
# Model and Data Loading (using integration module)
# ============================================================
def load_model_and_data(
    model_type: str = 'transplat',
    dataset_name: str = 're10k',
    checkpoint_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    evaluation_index: Optional[Path] = None,
    experiment_name: str = 're10k',
    hydra_overrides: tuple[str, ...] = (),
    device: Optional[torch.device] = None,
    num_samples: int = 1,
    sample_index: int = 0,
    calibration_target_free: bool = False,
):
    """
    Load model and data using the model loader abstraction.
    
    Args:
        model_type: Type of model ('transplat', 'mvsplat', 'depthsplat')
        num_samples: Number of test samples exposed by the dataloader
        sample_index: Zero-based sample index to select from the test dataloader
    
    Returns:
        (model, batch, cfg, device)
    """
    if checkpoint_path is None:
        checkpoint_path = SCARF_ROOT / model_type / 'checkpoints' / 're10k.ckpt'
    checkpoint_path = Path(checkpoint_path).resolve()
    
    # Check if checkpoint exists
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download the checkpoint for {model_type}.\n"
            f"See docs/multi-model-demo-guide.md for instructions."
        )
    
    cache_key = (
        model_type,
        str(checkpoint_path),
        str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        str(Path(evaluation_index).resolve()) if evaluation_index is not None else None,
        experiment_name,
        tuple(hydra_overrides),
        str(device) if device is not None else "auto",
    )
    cached = _MODEL_BUNDLE_CACHE.get(cache_key)
    if cached is None:
        loader = create_model_loader(model_type)
        model_bundle = loader.load_model(
            str(checkpoint_path),
            device=device,
            experiment_name=experiment_name,
            dataset_root=dataset_root,
            evaluation_index=evaluation_index,
            hydra_overrides=tuple(hydra_overrides),
        )
        _MODEL_BUNDLE_CACHE[cache_key] = (loader, model_bundle)
    else:
        loader, model_bundle = cached
    
    # Calibration uses a sidecar with context RGB and target camera geometry;
    # ordinary evaluation retains the model's native dataloader.
    if calibration_target_free:
        if dataset_root is None or evaluation_index is None:
            raise ValueError("target-free calibration requires dataset root and index")
        data_bundle = load_target_free_calibration_data(
            loader,
            model_bundle,
            dataset_name=dataset_name,
            dataset_root=Path(dataset_root),
            evaluation_index=Path(evaluation_index),
            sample_index=sample_index,
        )
    else:
        data_bundle = loader.load_data(
            model_bundle,
            dataset_name=dataset_name,
            num_samples=max(num_samples, sample_index + 1),
            sample_index=sample_index,
        )
    
    return model_bundle.model, data_bundle.batch, model_bundle.config, model_bundle.device


def tune_thresholds(
    gaussians_full,
    model, tgt_ext, tgt_int, target, h, w, device,
    gt_image, baseline_psnr,
    features, depths,
    quality_budget_pct: float = 0.2,
):
    """
    Sweep SAES v4 thresholds (L0+L1 only) to find the most aggressive
    settings within the quality budget.

    Sweeps: feature_var_threshold (L0), depth_std_threshold (L1).
    FSDR uses fixed hardware criteria.

    Args:
        quality_budget_pct: Max allowed relative PSNR loss in %

    Returns:
        (best_feat_var_threshold, best_fsdr_tolerance, saes_results, fsdr_results)
    """
    from src.model.types import Gaussians

    max_loss_db = baseline_psnr * quality_budget_pct / 100.0

    def compute_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        return -10 * torch.log10(mse).item()

    # ---- Phase 1: SAES v4 sweep (feature_var + depth_std) ----
    feat_var_values = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]
    depth_std_values = [0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]

    print("\n### Threshold Tuning: SAES v4 (L0+L1, Phase 1)")
    print(f"  Quality budget: {quality_budget_pct:.1f}% relative PSNR = {max_loss_db:.4f} dB")
    print(f"  {'FeatVar':>8s}  {'DepthStd':>9s}  {'L0%':>5s}  {'L1%':>5s}  "
          f"{'ModPx%':>7s}  {'PSNR':>8s}  {'Loss':>8s}  {'St':>3s}")

    best_combo = None
    best_mod_ratio = 0.0
    saes_results = []

    for fv in feat_var_values:
        for ds in depth_std_values:
            trial_g = Gaussians(
                means=gaussians_full.means.clone(),
                covariances=gaussians_full.covariances.clone(),
                harmonics=gaussians_full.harmonics.clone(),
                opacities=gaussians_full.opacities.clone(),
            )
            _, stats, _ = apply_progressive_saes(
                trial_g, h, w, CONFIG.tile_size, gpp=1,
                feature_var_threshold=fv,
                depth_std_threshold=ds, features=features, depths=depths,
                cross_check_threshold=CONFIG.saes_cross_check,
                beta_x=CONFIG.beta_x,
                beta_f=CONFIG.beta_f,
                beta_d=CONFIG.beta_d,
                num_depth_candidates=CONFIG.num_depth_candidates)

            with torch.no_grad():
                out = model.decoder.forward(
                    trial_g, tgt_ext, tgt_int,
                    target['near'], target['far'], (h, w), depth_mode=None
                )
            psnr = compute_psnr(out.color[0, 0], gt_image)
            loss_db = psnr - baseline_psnr
            loss_pct = abs(loss_db) / baseline_psnr * 100
            within_budget = loss_pct <= quality_budget_pct

            mod_ratio = stats.get('modification_ratio', 0.0)
            l0r = stats.get('level0_ratio', 0.0)
            l1r = stats.get('level1_ratio', 0.0)

            saes_results.append({
                'feat_var': fv, 'depth_std': ds,
                'psnr': psnr, 'loss_db': loss_db, 'loss_pct': loss_pct,
                'within_budget': within_budget, 'mod_ratio': mod_ratio,
                'l0_ratio': l0r, 'l1_ratio': l1r,
            })

            st = "OK" if within_budget else "X"
            if within_budget and mod_ratio > 0.05:
                print(f"  {fv:>8.3f}  {ds:>9.3f}  "
                      f"{l0r*100:>5.1f}  {l1r*100:>5.1f}  "
                      f"{mod_ratio*100:>7.1f}  {psnr:>8.4f}  {loss_db:>+8.4f}  {st:>3s}")

            if within_budget and mod_ratio > best_mod_ratio:
                best_mod_ratio = mod_ratio
                best_combo = (fv, ds, psnr, loss_db, mod_ratio)

    if best_combo:
        best_fv, best_ds, best_psnr, best_loss, best_mod = best_combo
        print(f"\n  ** Best SAES v4: feat_var={best_fv}, depth_std={best_ds}")
        print(f"     -> {best_mod*100:.1f}% pixels modified, "
              f"PSNR={best_psnr:.4f} dB, loss={best_loss:+.4f} dB")
    else:
        best_fv = 0.010
        best_ds = 0.005
        print(f"\n  ** No combo within budget, using defaults")
    
    # ---- Phase 2: FSDR reuse rate (realistic model) ----
    # In the realistic model, FSDR reuse criteria are hardware-fixed (hamming, spatial, confidence).
    # We report the reuse rate for reference — FSDR now has quality impact.
    print(f"\n### FSDR Reuse Rate (Realistic ASIC Model)")
    print(f"  (FSDR uses cached Gaussians → has quality impact)")
    print(f"  Reuse criteria: hamming≤{CONFIG.fsdr_reuse_hamming}, "
          f"spatial≤{CONFIG.fsdr_reuse_spatial}, conf>{CONFIG.fsdr_reuse_confidence}")
    
    fsdr_results = []
    N = gaussians_full.means.shape[1]
    
    has_features = (features is not None and
                   not isinstance(features, str) and
                   hasattr(features, 'shape'))
    
    if has_features:
        from scripts.fsdr_trace import prepare_fsdr_frame, tile_probe_pixel_order

        depth_input = depths
        if depth_input is None:
            depth_input = torch.ones(1, 1, h, w, device=features.device)
        fsdr_features, frame_depths = prepare_fsdr_frame(
            features, depth_input, height=h, width=w
        )
    
    trial_fsdr = FSDRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold,
        reuse_hamming=CONFIG.fsdr_reuse_hamming,
        reuse_spatial=CONFIG.fsdr_reuse_spatial,
        reuse_confidence=CONFIG.fsdr_reuse_confidence,
        num_depth_candidates=CONFIG.num_depth_candidates,
        seed=CONFIG.fsdr_seed,
        depth_consistency_threshold=CONFIG.gamma_depth,
        tile_size=CONFIG.tile_size,
    )
    
    if has_features:
        trial_fsdr.process_frame(
            fsdr_features,
            frame_depths,
            w,
            pixel_order=tile_probe_pixel_order(
                height=h, width=w, tile_size=CONFIG.tile_size
            ),
        )
    
    reuse_rate = trial_fsdr.get_reuse_ratio()
    print(f"  Reuse rate (skip S2+S3): {reuse_rate*100:.1f}%")
    print(f"  Cache hit rate: {trial_fsdr.stats['cache_hits']/max(1,trial_fsdr.stats['total_pixels'])*100:.1f}%")
    
    fsdr_results.append({
        'reuse_rate': reuse_rate,
        'hit_rate': trial_fsdr.stats['cache_hits'] / max(1, trial_fsdr.stats['total_pixels']),
    })
    
    best_fsdr_tol = 0.0  # Not applicable in realistic model
    
    print(f"\n  Best SAES v4: feat_var={best_fv}, depth_std={best_ds}")
    
    # Apply the multi-level thresholds to CONFIG
    CONFIG.feature_var_threshold = best_fv
    CONFIG.depth_std_threshold = best_ds

    return best_fv, best_fsdr_tol, saes_results, fsdr_results


def compute_saes_low_var_agreement(
    gaussians_full,
    modified_mask: torch.Tensor,
    h: int,
    w: int,
    tile_size: int,
    threshold: float = 0.90,
    view_count: int = 1,
) -> Dict[str, float]:
    """
    Check whether SAES early materialization is applied to tiles whose full
    per-pixel Gaussian outputs are already low-variation.

    The statistic is computed after a normal SAES run. A tile is considered an
    SAES L0/L1 tile if at least one non-probe pixel was modified by SAES. For
    each such tile, we measure the similarity of full per-pixel Gaussian
    attributes before SAES modification. High agreement means SAES selected
    tiles where the pretrained adaptor already produced locally similar outputs.
    """
    if modified_mask is None or modified_mask.numel() == 0:
        return {
            'early_tiles': 0,
            'low_var_tiles': 0,
            'low_var_agree': 0.0,
            'mean_similarity': 0.0,
            'threshold': threshold,
        }

    means = gaussians_full.means[0]
    covs = gaussians_full.covariances[0]
    harmo = gaussians_full.harmonics[0]
    opacs = gaussians_full.opacities[0].reshape(-1)
    n_gauss = means.shape[0]
    position_count = view_count * h * w
    if n_gauss % position_count != 0:
        raise ValueError("Gaussian layout is incompatible with SAES view count")
    primitives_per_pixel = n_gauss // position_count

    def _mean_pair_cos(x: torch.Tensor) -> float:
        x = x.reshape(x.shape[0], -1).float()
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = x @ x.t()
        n = sim.shape[0]
        if n < 2:
            return 1.0
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=sim.device), diagonal=1)
        return float(((sim[mask] + 1.0) * 0.5).mean().item())

    def _tile_similarity(indices: torch.Tensor) -> float:
        if indices.numel() < 2:
            return 1.0
        m = means[indices].float()
        pos_std = m.std(dim=0).mean()
        pos_range = m.abs().max() + 1e-8
        pos_sim = 1.0 - torch.clamp(pos_std / pos_range, 0.0, 1.0)

        cov_sim = _mean_pair_cos(covs[indices])
        sh_sim = _mean_pair_cos(harmo[indices])

        op = opacs[indices].float()
        n = op.shape[0]
        op_diff = torch.abs(op[:, None] - op[None, :])
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=op.device), diagonal=1)
        op_sim = 1.0 - torch.clamp(op_diff[mask].mean(), 0.0, 1.0)

        sim = 0.25 * float(pos_sim.item()) + 0.30 * cov_sim + 0.30 * sh_sim + 0.15 * float(op_sim.item())
        return float(sim)

    early_tiles = 0
    low_var_tiles = 0
    sims: List[float] = []

    tiles_h = h // tile_size
    tiles_w = w // tile_size
    for view in range(view_count):
        for th in range(tiles_h):
            for tw in range(tiles_w):
                tile_indices = []
                for ly in range(tile_size):
                    for lx in range(tile_size):
                        pixel = (th * tile_size + ly) * w + (tw * tile_size + lx)
                        for slot in range(primitives_per_pixel):
                            tile_indices.append(
                                (view * h * w + pixel) * primitives_per_pixel + slot
                            )
                idx_tensor = torch.tensor(
                    tile_indices, dtype=torch.long, device=modified_mask.device
                )
                if not bool(modified_mask[idx_tensor].any().item()):
                    continue

                early_tiles += 1
                sim = _tile_similarity(idx_tensor)
                sims.append(sim)
                if sim >= threshold:
                    low_var_tiles += 1

    mean_similarity = float(sum(sims) / len(sims)) if sims else 0.0
    low_var_agree = low_var_tiles / early_tiles if early_tiles > 0 else 0.0
    return {
        'early_tiles': early_tiles,
        'low_var_tiles': low_var_tiles,
        'low_var_agree': low_var_agree,
        'mean_similarity': mean_similarity,
        'threshold': threshold,
    }


# ============================================================
# Main Pipeline
# ============================================================
def main(argv=None):
    global CONFIG
    
    args = parse_demo_args(argv)
    strict_run = args.claim_run or args.functional_run or args.diagnostic_run
    from scripts.mechanism_config import (
        load_mechanism_config,
        require_calibrated_mechanism,
    )

    mechanism_config, mechanism_provenance = (
        require_calibrated_mechanism()
        if args.claim_run
        else load_mechanism_config()
    )
    fallback_stages = []
    from scripts.ae_config import (
        resolve_experiment,
        validate_claim_dataset_tree,
        validate_prepared_dataset,
    )
    from scripts.runtime_assets import validate_runtime_assets

    experiment = resolve_experiment(args.model, args.dataset, SCARF_ROOT)
    runtime_assets = validate_runtime_assets(args.model)
    checkpoint_path = (args.checkpoint or experiment.checkpoint).resolve()
    dataset_root = (args.dataset_root or experiment.dataset_root).resolve()
    dataset_manifest = (
        args.dataset_manifest or dataset_root / ".scarf-manifest.json"
    ).resolve()
    if args.calibration_trace:
        from scripts.calibration_inputs import validate_target_free_input_root

        dataset_identity = validate_target_free_input_root(dataset_root, args.dataset)
    else:
        if not dataset_manifest.is_file():
            raise FileNotFoundError(
                f"dataset provenance manifest not found: {dataset_manifest}\n"
                "Run the documented dataset preparation command before evaluation."
            )
        dataset_identity = validate_prepared_dataset(
            experiment,
            dataset_root,
            dataset_manifest,
            allow_functional_fixture=args.functional_run,
        )
    if args.claim_run:
        validate_claim_dataset_tree(
            args.model,
            args.dataset,
            dataset_identity['tree_sha256'],
            SCARF_ROOT,
        )
    if args.device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(args.device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {args.device}")

    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Override global SCARF frequency from CLI
    global SCARF_FREQ_MHZ
    SCARF_FREQ_MHZ = args.freq

    # --ablation implies both SAES and FSDR must run
    if args.ablation:
        if args.no_saes:
            print("[ablation] Overriding --no-saes: SAES will run for ablation data")
            args.no_saes = False
        if args.no_fsdr:
            print("[ablation] Overriding --no-fsdr: FSDR will run for ablation data")
            args.no_fsdr = False
    
    # Initialize config
    CONFIG = SCARFConfig(model_type=args.model)
    engineering = mechanism_config.get('selected') or {
        'gamma_depth': CONFIG.gamma_depth,
        'beta_x': CONFIG.beta_x,
        'beta_f': CONFIG.beta_f,
        'beta_d': CONFIG.beta_d,
    }
    CONFIG.gamma_depth = float(engineering['gamma_depth'])
    CONFIG.beta_x = float(engineering['beta_x'])
    CONFIG.beta_f = float(engineering['beta_f'])
    CONFIG.beta_d = float(engineering['beta_d'])

    # Apply CLI threshold overrides (after CONFIG is initialized)
    if args.saes_fv is not None:
        CONFIG.feature_var_threshold = args.saes_fv
    if args.saes_ds is not None:
        CONFIG.depth_std_threshold = args.saes_ds
    if args.saes_cc is not None:
        CONFIG.saes_cross_check = args.saes_cc
    if args.tile_size is not None:
        CONFIG.tile_size = args.tile_size
    if args.fsdr_cache_size is not None:
        CONFIG.fsdr_cache_size = args.fsdr_cache_size
    if args.fsdr_hamming is not None:
        CONFIG.fsdr_hamming_threshold = args.fsdr_hamming
        CONFIG.fsdr_reuse_hamming = args.fsdr_hamming

    # Create adapter and apply model-specific overrides
    adapter = create_adapter(args.model)
    CONFIG.apply_adapter_overrides(adapter)

    # Keep explicit experiment overrides highest priority.
    if args.saes_fv is not None:
        CONFIG.feature_var_threshold = args.saes_fv
    if args.saes_ds is not None:
        CONFIG.depth_std_threshold = args.saes_ds
    if args.saes_cc is not None:
        CONFIG.saes_cross_check = args.saes_cc
    if args.tile_size is not None:
        CONFIG.tile_size = args.tile_size
    if args.fsdr_cache_size is not None:
        CONFIG.fsdr_cache_size = args.fsdr_cache_size
    if args.fsdr_hamming is not None:
        CONFIG.fsdr_hamming_threshold = args.fsdr_hamming
        CONFIG.fsdr_reuse_hamming = args.fsdr_hamming
    
    # All models use the same universal SAES/FSDR configuration.
    # DINOv2 (DepthSplat) vs CNN (TranSplat/MVSplat) features naturally have different
    # variance distributions, but the threshold is set to work well across all models.
    
    print("=" * 70)
    print(f"SCARF Demo - Hardware Simulator for 3DGS Encoders")
    print(f"Model: {args.model}")
    print("=" * 70)
    print()
    print(f"[Config] SAES v2:")
    print(f"  tile={CONFIG.tile_size}, feat_var={CONFIG.feature_var_threshold}, depth_std={CONFIG.depth_std_threshold}")
    print(f"  L0+L1 probe-anchored Gaussian moment matching")
    print(f"[Config] FSDR (realistic ASIC):")
    print(f"  cache_size={CONFIG.fsdr_cache_size}, hamming_threshold={CONFIG.fsdr_hamming_threshold}")
    print(f"  guidance_policy={args.fsdr_guidance_policy}")
    print(f"  gamma_depth={CONFIG.gamma_depth}")
    print(f"[Config] GGU:")
    print(f"  scale_range=({CONFIG.scale_min}, {CONFIG.scale_max})")
    print()
    
    # Initialize cycle counter and savings tracker
    cycle_counter = HWCycleCounter(ggu_pe_count=CONFIG.ggu_pe_count)
    savings = SavingsTracker(model_type=args.model)
    
    # --------------------------------------------------------
    # Step 1: Load Model and Data
    # --------------------------------------------------------
    print("[1/6] Loading model and data...")
    model, batch, cfg, device = load_model_and_data(
        args.model,
        dataset_name=args.dataset,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        evaluation_index=args.evaluation_index,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=selected_device,
        num_samples=args.num_samples,
        sample_index=args.sample_index,
        calibration_target_free=args.calibration_trace,
    )
    print("  ✓ Model and data loaded")
    model_depth_predictor = getattr(model.encoder, 'depth_predictor', None)
    actual_depth_candidates = getattr(
        model_depth_predictor, 'num_depth_candidates', None
    )
    if (
        not isinstance(actual_depth_candidates, int)
        or isinstance(actual_depth_candidates, bool)
        or actual_depth_candidates < 4
        or actual_depth_candidates % 4
    ):
        raise RuntimeError(
            "loaded depth predictor does not expose a valid D divisible by four"
        )
    CONFIG.num_depth_candidates = actual_depth_candidates
    CONFIG.fsdr_narrowed_candidates = actual_depth_candidates // 4
    print(
        "  ✓ S2 candidates from loaded predictor: "
        f"D={CONFIG.num_depth_candidates}, narrowed={CONFIG.fsdr_narrowed_candidates}"
    )
    
    # --------------------------------------------------------
    # Step 2: Extract batch info
    # --------------------------------------------------------
    print()
    print("[2/6] Processing batch...")
    
    B, V_ctx, _, h, w = batch['context']['image'].shape
    native_dataloader_loaded_target_rgb = False
    target_rgb_removed_before_execution = False
    target_rgb_free_execution = is_target_rgb_free_execution(args)
    if args.calibration_trace:
        if 'image' in batch['target']:
            raise RuntimeError("calibration batch unexpectedly contains target RGB")
        V_tgt = int(batch['target']['extrinsics'].shape[1])
    elif target_rgb_free_execution:
        # The ordinary official dataloader may have loaded target RGB. Remove
        # it before device transfer or any model/audit invocation so this
        # diagnostic cannot accidentally use RGB evidence.
        V_tgt = int(batch['target']['extrinsics'].shape[1])
        native_dataloader_loaded_target_rgb = 'image' in batch['target']
        batch['target'].pop('image', None)
        target_rgb_removed_before_execution = True
    else:
        _, V_tgt, _, _, _ = batch['target']['image'].shape
    scene_name = batch['scene'][0] if 'scene' in batch else 'unknown'
    
    print(f"  ✓ Scene: {scene_name}, Testing {V_tgt} view(s), Size: {h}x{w}")
    
    context = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch['context'].items()}
    target = {
        k: v[:, :V_tgt].to(device) if torch.is_tensor(v) and v.dim() > 1 else (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch['target'].items()
        if not (target_rgb_free_execution and k == 'image')
    }
    
    # --------------------------------------------------------
    # Step 3: Run BASELINE (full encoder) when image evidence needs it.
    # --------------------------------------------------------
    print()
    gpu_info = get_gpu_info()
    gpu_freq_mhz = gpu_info['freq_mhz']
    gpu_name = gpu_info['name']
    paired_rng_state = capture_torch_rng_state()
    if args.fsdr_only:
        print("[3/6] Baseline encoder and renderer: SKIPPED (--fsdr-only)")
        print(f"  GPU: {gpu_name}")
    elif (
        args.saes_s3_raw_audit
        or args.saes_routing_audit
        or args.saes_hardware_audit
    ):
        audit_name = (
            "S3 raw audit"
            if args.saes_s3_raw_audit
            else "SAES routing audit"
            if args.saes_routing_audit
            else "SAES hardware audit"
        )
        print(f"[3/6] Baseline encoder and renderer: SKIPPED ({audit_name})")
        if args.saes_hardware_audit:
            print("  The diagnostic records S1--S4 events but never renders or scores images.")
        else:
            print("  The diagnostic stops before GGU/S4 and rendering.")
    else:
        print("[3/6] Running BASELINE (original model)...")
        print(f"  GPU: {gpu_name} @ {gpu_freq_mhz} MHz (max SM clock)")
        with torch.no_grad():
            encoder_output = model.encoder(context, False, deterministic=False)
            if isinstance(encoder_output, dict):
                baseline_gaussians = encoder_output.get('gaussians', encoder_output)
            else:
                baseline_gaussians = encoder_output
            tgt_ext = target['extrinsics']
            tgt_int = target['intrinsics']
            baseline_output = model.decoder.forward(
                baseline_gaussians,
                tgt_ext,
                tgt_int,
                target['near'],
                target['far'],
                (h, w),
                depth_mode=None,
            )
        baseline_images = baseline_output.color[0, :V_tgt]
        baseline_image = baseline_images[0]
        baseline_count = baseline_gaussians.means.shape[1]

        def _baseline_encoder():
            return model.encoder(context, False, deterministic=False)

        if args.sensitivity_trace:
            baseline_gpu_time_ms = 0.0
            baseline_timing_samples_ms = []
        else:
            is_orin = device.type == "cuda" and "Orin" in gpu_name
            timing_repetitions = 5 if is_orin else 1
            baseline_gpu_time_ms, baseline_timing_samples_ms = gpu_timed_inference(
                _baseline_encoder,
                warmup=1 if is_orin else 0,
                repeats=timing_repetitions,
                device=device,
                return_samples=True,
            )
        baseline_gpu_freq_hz = gpu_freq_mhz * 1e6
        baseline_gpu_cycles = int(
            baseline_gpu_time_ms * 1e-3 * baseline_gpu_freq_hz
        )
        print(f"  ✓ Baseline: {baseline_image.shape}, Gaussians: {baseline_count:,}")
        print(
            f"    GPU encoder time: {baseline_gpu_time_ms:.2f} ms "
            f"({len(baseline_timing_samples_ms)} diagnostic sample(s))"
        )
        print(f"    GPU encoder cycles: {baseline_gpu_cycles:,} (@ {gpu_freq_mhz} MHz)")
        if hasattr(baseline_gaussians, 'depths'):
            baseline_depths = baseline_gaussians.depths.reshape(-1)
            print(
                "    Baseline depth stats: "
                f"min={baseline_depths.min():.4f}, "
                f"max={baseline_depths.max():.4f}, "
                f"mean={baseline_depths.mean():.4f}"
            )
        elif hasattr(baseline_gaussians, 'means'):
            z_depths = baseline_gaussians.means.reshape(-1, 3)[:, 2]
            print(
                "    Baseline Z-coord depth: "
                f"min={z_depths.min():.4f}, max={z_depths.max():.4f}, "
                f"mean={z_depths.mean():.4f}"
            )
        if hasattr(baseline_gaussians, 'opacities'):
            bl_op = baseline_gaussians.opacities.reshape(-1)
            print(
                "    Baseline opacities: "
                f"min={bl_op.min():.4f}, max={bl_op.max():.4f}, "
                f"mean={bl_op.mean():.4f}"
            )
        if args.baseline_only:
            gt_image = target['image'][0, 0]
            mse = F.mse_loss(baseline_image, gt_image)
            psnr = -10 * torch.log10(mse).item()
            print()
            print("[4/6] SCARF Pipeline: SKIPPED (--baseline-only)")
            print("[5/6] Rendering: SKIPPED (--baseline-only)")
            print()
            print("=" * 70)
            print("RESULTS - Baseline Only")
            print("=" * 70)
            print()
            print("### Baseline Metrics")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  Gaussians: {baseline_count:,}")
            print(
                f"  GPU encoder time: {baseline_gpu_time_ms:.2f} ms "
                f"({len(baseline_timing_samples_ms)} diagnostic sample(s))"
            )
            print(
                f"  GPU encoder cycles: {baseline_gpu_cycles:,} "
                f"(@ {gpu_freq_mhz} MHz)"
            )
            return
    
    # --------------------------------------------------------
    # Step 4: Run SCARF Pipeline (Clear 3-Stage Flow)
    # --------------------------------------------------------
    # Pipeline Architecture:
    #   Stage 1: Feature Extraction → pipeline_features
    #   Stage 2: Depth Prediction   → pipeline_depths, pipeline_densities, pipeline_raw_gaussians
    #   Stage 3: Gaussian Generation → scarf_gaussians_full
    #
    # All stages use SCARF hardware simulators by default:
    #   S1 (--no-feature), S2 (--no-depth), S3 (--no-gaussian)
    # When a stage is disabled, use original GPU computation as fallback
    # --------------------------------------------------------
    print()
    print("[4/6] Running SCARF Pipeline...")
    
    t0 = time.time()
    from encoder.mmcu_events import mmcu_stage, mmcu_stage_records, reset_mmcu_events

    reset_mmcu_events()
    
    # Initialize cycle counters
    feature_sim_cycles = 0
    s1_cnn_cycles = 0    # CNN portion of S1 (for per-component ASIC scaling)
    depth_sim_cycles = 0
    dp_core_cycles = 0   # DP core: cost_volume + unet + depth_head + regression
    cost_volume_cycles = 0  # cost_volume portion of dp_core (for bandwidth modeling)
    gauss_gen_cycles = 0  # Gaussian Gen: refine_unet + to_gaussians (full-res)
    
    # ============================================================
    # Common depth range (needed by DepthSplat paths in both S1 and S2)
    # ============================================================
    near = context.get('near', target.get('near', torch.tensor([0.5], device=device)))
    far = context.get('far', target.get('far', torch.tensor([100.0], device=device)))
    if near.dim() == 0:
        near = near.unsqueeze(0)
    if far.dim() == 0:
        far = far.unsqueeze(0)
    
    # ============================================================
    # STAGE 1: Feature Extraction
    # ============================================================
    print("  [Stage 1] Feature Extraction...")
    
    # Pipeline variables for Stage 1 output
    pipeline_features = None
    pipeline_cnn_features = None
    pipeline_mono_features = None
    pipeline_saes_features = None
    depthsplat_results = None  # Store DepthSplat results_dict for S3
    
    use_feature_sim = not args.no_feature
    
    if use_feature_sim:
        # Use SCARF hardware simulator for feature extraction
        print(f"    Mode: Hardware Simulator")
        from feature_extractor import (
            TransplatFeatureExtractor,
            MVSplatFeatureExtractor,
            DepthSplatFeatureExtractor,
        )
        
        try:
            if args.model == 'transplat':
                feature_extractor = TransplatFeatureExtractor.from_encoder(model.encoder)
            elif args.model == 'mvsplat':
                feature_extractor = MVSplatFeatureExtractor.from_encoder(model.encoder)
            elif args.model == 'depthsplat':
                feature_extractor = DepthSplatFeatureExtractor.from_encoder(model.encoder)
            else:
                feature_extractor = None
            
            if feature_extractor is not None:
                with mmcu_stage("s1"):
                    fe_output = feature_extractor.forward(
                        context['image'],
                        context.get('extrinsics'),
                        context.get('intrinsics'),
                    )
                feature_sim_cycles = fe_output.total_cycles
                
                # Extract CNN cycle count for per-component ASIC scaling.
                # Note: fe_output is one of {TranSplat,MVSplat,DepthSplat}FeatureOutput
                # — different dataclasses with slightly different fields (e.g. only
                # DepthSplat has dinov2_cycles). getattr() handles this polymorphism.
                s1_cnn_cycles = getattr(fe_output, 'cnn_cycles', 0)
                
                # Print cycle breakdown
                cycle_info = []
                if s1_cnn_cycles > 0:
                    cycle_info.append(f"CNN: {s1_cnn_cycles:,}")
                trans_c = getattr(fe_output, 'transformer_cycles', 0)
                if trans_c > 0:
                    cycle_info.append(f"Transformer: {trans_c:,}")
                dino_c = getattr(fe_output, 'dinov2_cycles', 0)
                if dino_c > 0:
                    cycle_info.append(f"DINOv2: {dino_c:,}")
                
                print(f"    ✓ HW cycles: {feature_sim_cycles:,} ({', '.join(cycle_info)})")
                
                # Extract features from simulator output
                _trans = getattr(fe_output, 'trans_features', None)
                if _trans is not None:
                    pipeline_features = _trans
                _cnn = getattr(fe_output, 'cnn_features', None)
                if _cnn is not None:
                    pipeline_cnn_features = _cnn
                _mono = getattr(fe_output, 'mono_features', None)
                if _mono is not None:
                    pipeline_mono_features = _mono
                
                print(f"    ✓ Features: {pipeline_features.shape if pipeline_features is not None else 'None'}")
        except Exception as e:
            if strict_run:
                raise strict_stage_error("feature simulator", e) from e
            print(f"    ⚠ HW Simulator error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_feature_sim = False
    
    if not use_feature_sim or pipeline_features is None:
        if strict_run:
            raise RuntimeError(
                "strict run feature simulator produced no features; GPU fallback is forbidden"
            )
        fallback_stages.append("feature")
        # Fallback: Use original GPU for feature extraction
        print(f"    Mode: Original GPU" + (" (fallback)" if use_feature_sim else " (--no-feature)"))
        
        with torch.no_grad():
            if args.model == 'depthsplat':
                # DepthSplat: depth_predictor (MultiViewUniMatch) does S1+S2 internally
                # Call it directly under SCARF control (no monolithic encoder() call)
                b_ds, v_ds, _, h_ds, w_ds = context['image'].shape
                near_bv = near.expand(b_ds, v_ds) if near.dim() <= 1 else near
                far_bv = far.expand(b_ds, v_ds) if far.dim() <= 1 else far
                near_bv = near_bv.to(device).clamp(min=1e-6)
                far_bv = far_bv.to(device).clamp(min=1e-6)
                
                depthsplat_results = model.encoder.depth_predictor(
                    context['image'],
                    attn_splits_list=[2],
                    intrinsics=context['intrinsics'],
                    min_depth=1.0 / far_bv,
                    max_depth=1.0 / near_bv,
                    extrinsics=context['extrinsics'],
                )
                # Extract real feature tensor for FSDR/SAES
                pipeline_features = rearrange(
                    depthsplat_results['features_mv'][0],
                    '(b v) c h w -> b v c h w', b=b_ds, v=v_ds
                )
                pipeline_cnn_features = None
                print(f"    ✓ Features: {pipeline_features.shape}")
            elif hasattr(model.encoder, 'backbone'):
                if args.model == 'transplat':
                    from feature_extractor.transplat_extractor import (
                        transplat_image_to_world,
                    )

                    img2world = transplat_image_to_world(
                        context['extrinsics'],
                        context['intrinsics'],
                        h,
                        w,
                    ).contiguous()
                    pipeline_features, pipeline_cnn_features = model.encoder.backbone(
                        context['image'],
                        attn_splits=2,
                        return_cnn_features=True,
                        img2world=img2world,
                    )
                elif args.model == 'mvsplat':
                    pipeline_features, pipeline_cnn_features = model.encoder.backbone(
                        context['image'],
                        attn_splits=2,
                        return_cnn_features=True,
                    )
                else:
                    pipeline_features = None
                    pipeline_cnn_features = None
                print(f"    ✓ Features: {pipeline_features.shape if pipeline_features is not None else 'None'}")
            else:
                pipeline_features = None
                pipeline_cnn_features = None
                print(f"    ✓ Features: None")
    
    # ============================================================
    # STAGE 2: Depth Prediction
    # ============================================================
    print("  [Stage 2] Depth Prediction...")
    
    # Pipeline variables for Stage 2 output
    pipeline_depths = None
    pipeline_densities = None
    pipeline_raw_gaussians = None
    fsdr_depth_probs = None
    fsdr_depth_candidates = None
    fsdr_candidate_domain = None
    fsdr_probability_source = None
    
    # near/far already computed before Stage 1 (used by DepthSplat GPU fallback)
    
    # For TranSplat: compute da_depth and dino_feature (needed for both HW and GPU modes)
    dp_da_depth = None
    dp_dino_feature = None
    
    if args.model == 'transplat' and hasattr(model.encoder, 'da_model'):
        print(f"    Computing DINOv2 auxiliary features...")
        with torch.no_grad():
            b, v, c, h_img, w_img = context['image'].shape
            
            # Normalize images
            da_images = context['image'].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1).to(device)
            da_images = (da_images - mean) / std
            da_images = da_images[:, :, [2, 0, 1]]  # RGB -> BGR
            
            da_images_flat = da_images.view(b*v, c, h_img, w_img)
            da_images_resized = F.interpolate(da_images_flat, (252, 252), mode='bilinear', align_corners=True)
            
            # Call da_model
            da_depth_raw, out_feature = model.encoder.da_model.forward(da_images_resized)
            
            # Process da_depth
            dp_da_depth = F.interpolate(da_depth_raw[None], (h_img, w_img), mode='bilinear', align_corners=True)
            dp_da_depth = dp_da_depth.view(b, v, 1, h_img, w_img)
            da_depth_flat = dp_da_depth.flatten(2)
            da_max = torch.max(da_depth_flat, dim=-1, keepdim=True)[0]
            da_min = torch.min(da_depth_flat, dim=-1, keepdim=True)[0]
            dp_da_depth = (da_depth_flat - da_min) / (da_max - da_min + 1e-8)
            dp_da_depth = dp_da_depth.reshape(b, v, 1, h_img, w_img)
            
            # Process dino_feature
            dp_dino_feature = out_feature.view(b, v, out_feature.shape[1], out_feature.shape[2], out_feature.shape[3])
    
    use_depth_sim = not args.no_depth
    
    if use_depth_sim:
        # Use SCARF hardware simulator for depth prediction
        print(f"    Mode: Hardware Simulator")
        from depth_predictor import (
            TransplatDepthPredictorSim,
            MVSplatDepthPredictorSim,
            DepthSplatDepthPredictorSim,
        )
        
        try:
            depth_predictor_sim = None
            
            if args.model == 'transplat':
                depth_predictor_sim = TransplatDepthPredictorSim(device=device)
                depth_predictor_sim.load_from_model(model.encoder)
                depth_predictor_sim.set_accurate_mode(True)
            elif args.model == 'mvsplat':
                depth_predictor_sim = MVSplatDepthPredictorSim(device=device)
                depth_predictor_sim.load_from_model(model.encoder)
                depth_predictor_sim.set_accurate_mode(True)
            elif args.model == 'depthsplat':
                depth_predictor_sim = DepthSplatDepthPredictorSim(device=device)
                depth_predictor_sim.load_from_model(model.encoder)
            if depth_predictor_sim is not None:
                depth_predictor_sim.set_strict_mode(strict_run)
            
            # DepthSplat can work with either features or images (features extracted internally)
            can_run_hw = (depth_predictor_sim is not None and
                         (pipeline_features is not None or args.model == 'depthsplat'))
            
            if can_run_hw:
                extra_info = {'images': rearrange(context['image'], 'b v c h w -> (v b) c h w')}
                
                with torch.no_grad():
                    # The paper evaluates probabilistic Gaussian sampling. Replay the
                    # exact stream used by the baseline so the hardware cycle model
                    # changes execution cost, not the sampled neural result.
                    restore_torch_rng_state(paired_rng_state)
                    with mmcu_stage("s2"):
                        dp_output = depth_predictor_sim.forward(
                            pipeline_features,  # Input from Stage 1
                            context['intrinsics'],
                            context['extrinsics'],
                            near,
                            far,
                            images=context.get('image'),
                            da_depth=dp_da_depth,
                            dino_feature=dp_dino_feature,
                            cnn_features=pipeline_cnn_features,
                            extra_info=extra_info,
                            deterministic=False,
                        )
                
                cycle_breakdown = dp_output.cycle_breakdown.to_dict()
                cost_volume_cycles = cycle_breakdown.get('cost_volume', 0)
                
                # Separate S2 (depth prediction) from S3 (gaussian generation)
                # The HW depth predictor may include gaussian_head cycles
                gaussian_head_from_dp = cycle_breakdown.get('gaussian_head', 0)
                dp_core_cycles = dp_output.total_cycles - gaussian_head_from_dp
                depth_sim_cycles = dp_core_cycles
                
                # If depth predictor also produced S3 cycles, use those instead of fallback
                if gaussian_head_from_dp > 0:
                    gauss_gen_cycles = gaussian_head_from_dp
                    print(f"    ✓ S3 gaussian_head cycles from depth predictor: {gaussian_head_from_dp:,}")
                
                # DepthSplat's depth predictor re-runs feature extraction internally
                # (CNN backbone + MV Transformer + DINOv2 ViT). These are S1-shared
                # operations that only execute ONCE in hardware. Subtract the DP's
                # OWN reported feature_extraction cycles (not S1's cycle count).
                fe_in_dp = cycle_breakdown.get('feature_extraction', 0)
                if fe_in_dp > 0:
                    dp_core_cycles -= fe_in_dp
                    depth_sim_cycles = dp_core_cycles
                    print(f"    [DepthSplat] S2 de-duplicated: subtracted {fe_in_dp:,} "
                          f"DP-internal feature extraction cycles (S1-shared)")
                elif args.model == 'depthsplat':
                    raise RuntimeError(
                        "DepthSplat cycle breakdown has no feature_extraction field"
                    )
                
                # --- Probe-first scheduling overhead (Gap 5 / Dataflow spec) ---
                # Per the SCARF Dataflow spec, S2 processes pixels within each tile in
                # order Q_T = [C_T, P_T\C_T, R_T]: corner probes first, then non-corner
                # probes, then remaining pixels.  The SAES judgment signal is broadcast
                # to the tile's remaining pixel queue once all K(T) probes are complete.
                # This serialisation introduces a 1-cycle pipeline bubble per tile
                # (judgment_delay) between the last probe result and the first non-probe
                # dispatch decision.  We add this as an architectural overhead term.
                from saes import ProgressiveSAES as _SAES
                _K_T = len(_SAES.compute_probe_positions(CONFIG.tile_size))
                _n_tiles = (h // CONFIG.tile_size) * (w // CONFIG.tile_size)
                probe_queue_delay_cycles = _n_tiles  # 1 decision cycle per tile
                dp_core_cycles += probe_queue_delay_cycles
                depth_sim_cycles = dp_core_cycles
                print(f"    [S2] Probe-first scheduling: K(T)={_K_T} probes/tile, "
                      f"judgment_delay={probe_queue_delay_cycles:,} cycles "
                      f"({_n_tiles:,} tiles × 1 cy/tile)")

                print(f"    ✓ HW cycles: {dp_output.total_cycles:,} (S2: {dp_core_cycles:,}, "
                      f"FE_in_DP: {fe_in_dp:,}, S3_gauss_head: {gaussian_head_from_dp:,})")
                print(f"      Breakdown: feature_extraction={fe_in_dp:,}, "
                      f"cost_volume={cycle_breakdown.get('cost_volume', 0):,}, "
                      f"unet={cycle_breakdown.get('unet_refinement', 0):,}, "
                      f"depth_head={cycle_breakdown.get('depth_head', 0):,}, "
                      f"regression={cycle_breakdown.get('softmax_regression', 0):,}"
                      f"{', gauss_head=' + str(gaussian_head_from_dp) if gaussian_head_from_dp > 0 else ''}")
                
                # Extract outputs
                pipeline_depths = dp_output.depths
                pipeline_densities = dp_output.densities
                pipeline_raw_gaussians = dp_output.raw_gaussians
                pipeline_saes_features = dp_output.saes_features
                fsdr_depth_probs = dp_output.depth_probs
                fsdr_depth_candidates = dp_output.depth_candidates
                fsdr_candidate_domain = dp_output.candidate_domain
                fsdr_probability_source = dp_output.probability_source
                
                print(f"    ✓ Depths: {pipeline_depths.shape if pipeline_depths is not None else 'None'}")
                
                if pipeline_depths is not None:
                    d_flat = pipeline_depths.reshape(-1)
                    print(f"      Depth stats: min={d_flat.min():.4f}, max={d_flat.max():.4f}, mean={d_flat.mean():.4f}")
                
                # DepthSplat special case: depth_predictor doesn't return raw_gaussians
                # We need to run encoder's feature_upsampler, gaussian_regressor, gaussian_head
                if args.model == 'depthsplat' and pipeline_raw_gaussians is None and pipeline_depths is not None:
                    print(f"    [DepthSplat] Computing raw_gaussians from encoder modules...")
                    try:
                        with torch.no_grad():
                            b_ds, v_ds, _, h_ds, w_ds = context['image'].shape
                            near_bv = near.expand(b_ds, v_ds) if near.dim() <= 1 else near
                            far_bv = far.expand(b_ds, v_ds) if far.dim() <= 1 else far
                            near_bv = near_bv.to(device).clamp(min=1e-6)
                            far_bv = far_bv.to(device).clamp(min=1e-6)

                            # The hardware path above supplies cycle counts. Replay
                            # the baseline RNG before taking the pinned upstream
                            # numerical path used for quality and mechanism evidence.
                            restore_torch_rng_state(paired_rng_state)
                            results_dict = model.encoder.depth_predictor(
                                context['image'],
                                attn_splits_list=[2],
                                intrinsics=context['intrinsics'],
                                min_depth=1.0 / far_bv,
                                max_depth=1.0 / near_bv,
                                extrinsics=context['extrinsics'],
                            )
                            depthsplat_results = results_dict
                            from scripts.depthsplat_execution import (
                                extract_depthsplat_execution_tensors,
                            )
                            from scripts.fsdr_trace import (
                                depthsplat_global_candidate_tensors,
                            )

                            reference_tensors = extract_depthsplat_execution_tensors(
                                results_dict,
                                batch_size=b_ds,
                                view_count=v_ds,
                                image_height=h_ds,
                                image_width=w_ds,
                            )
                            # The Gaussian head below is exact upstream execution.
                            # Do not combine it with approximate hardware depth or
                            # feature tensors: that would corrupt ASIC-no-opt before
                            # SAES/FSDR are applied.
                            pipeline_depths = reference_tensors.depths
                            pipeline_densities = reference_tensors.densities
                            pipeline_features = reference_tensors.matching_features
                            pipeline_mono_features = reference_tensors.mono_features

                            fsdr_depth_probs, fsdr_depth_candidates = (
                                depthsplat_global_candidate_tensors(
                                    results_dict['match_probs'],
                                    near=near_bv,
                                    far=far_bv,
                                )
                            )
                            fsdr_candidate_domain = 'inverse_depth'
                            fsdr_probability_source = (
                                'pinned_original_depthsplat_first_scale_softmax'
                            )
                            
                            depth_final = reference_tensors.final_depth
                            match_prob = reference_tensors.final_match_probability

                            from depth_predictor.module_cycle_trace import (
                                run_module_with_cycle_trace,
                            )

                            feature_upsampler_trace = run_module_with_cycle_trace(
                                model.encoder.feature_upsampler,
                                results_dict["features_mono_intermediate"],
                                cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                mv_features=results_dict["features_mv"][0] if model.encoder.cfg.num_scales == 1
                                           else results_dict["features_mv"][::-1],
                                mmcu_stage_name="s3",
                            )
                            features_upsampled = feature_upsampler_trace.output
                            
                            match_prob_max = torch.max(match_prob, dim=1, keepdim=True)[0]
                            if match_prob_max.shape[-2:] != depth_final.shape[-2:]:
                                match_prob_max = F.interpolate(match_prob_max, size=depth_final.shape[-2:], mode='nearest')
                            
                            concat_input = torch.cat((
                                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                                rearrange(depth_final, "b v h w -> (b v) () h w"),
                                match_prob_max,
                                features_upsampled,
                            ), dim=1)
                            
                            gaussian_regressor_trace = run_module_with_cycle_trace(
                                model.encoder.gaussian_regressor,
                                concat_input,
                                mmcu_stage_name="s3",
                            )
                            regressor_out = gaussian_regressor_trace.output
                            
                            gaussian_head_input = torch.cat([
                                regressor_out,
                                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                                features_upsampled,
                                match_prob_max,
                            ], dim=1)
                            from depth_predictor.types import align_batch_major_features

                            pipeline_saes_features = align_batch_major_features(
                                gaussian_head_input,
                                batch_size=b_ds,
                                view_count=v_ds,
                            )
                            
                            gaussian_head_trace = run_module_with_cycle_trace(
                                model.encoder.gaussian_head,
                                gaussian_head_input,
                                mmcu_stage_name="s3",
                            )
                            gaussians_bv = gaussian_head_trace.output
                            gauss_gen_cycles = (
                                feature_upsampler_trace.total_cycles
                                + gaussian_regressor_trace.total_cycles
                                + gaussian_head_trace.total_cycles
                            )
                            pipeline_raw_gaussians = gaussians_bv
                            print(
                                "      ✓ Numeric source rebound to pinned "
                                "DepthSplat cost-volume/Gaussian execution "
                                f"({pipeline_features.shape[-2]}x"
                                f"{pipeline_features.shape[-1]} matching scale)"
                            )
                            print(f"      ✓ raw_gaussians computed: {pipeline_raw_gaussians.shape}")
                            print(
                                "      ✓ S3 traced cycles: "
                                f"{gauss_gen_cycles:,} "
                                f"(feature upsampler={feature_upsampler_trace.total_cycles:,}, "
                                f"regressor={gaussian_regressor_trace.total_cycles:,}, "
                                f"head={gaussian_head_trace.total_cycles:,})"
                            )
                    except Exception as e2:
                        if strict_run:
                            raise strict_stage_error(
                                "DepthSplat Gaussian input generation", e2
                            ) from e2
                        print(f"      ⚠ Failed to compute raw_gaussians: {e2}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            if strict_run:
                raise strict_stage_error("depth simulator", e) from e
            print(f"    ⚠ HW Simulator error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_depth_sim = False
    
    if not use_depth_sim or pipeline_depths is None:
        if strict_run:
            raise RuntimeError(
                "strict run depth simulator produced no depths; GPU fallback is forbidden"
            )
        fallback_stages.append("depth")
        # Fallback: Use original GPU for depth prediction
        print(f"    Mode: Original GPU" + (" (fallback)" if use_depth_sim else " (--no-depth)"))
        
        if pipeline_features is not None and not isinstance(pipeline_features, str) and hasattr(model.encoder, 'depth_predictor'):
            extra_info = {'images': rearrange(context['image'], 'b v c h w -> (v b) c h w')}
            
            with torch.no_grad():
                restore_torch_rng_state(paired_rng_state)
                if args.model == 'transplat':
                    pipeline_depths, pipeline_densities, pipeline_raw_gaussians = model.encoder.depth_predictor(
                        pipeline_features,
                        context['intrinsics'],
                        context['extrinsics'],
                        near,
                        far,
                        gaussians_per_pixel=1,
                        deterministic=False,
                        extra_info=extra_info,
                        cnn_features=pipeline_cnn_features,
                        da_depth=dp_da_depth,
                        dino_feature=dp_dino_feature,
                    )
                elif args.model == 'mvsplat':
                    pipeline_depths, pipeline_densities, pipeline_raw_gaussians = model.encoder.depth_predictor(
                        pipeline_features,
                        context['intrinsics'],
                        context['extrinsics'],
                        near,
                        far,
                        gaussians_per_pixel=1,
                        deterministic=False,
                        extra_info=extra_info,
                        cnn_features=pipeline_cnn_features,
                    )
            
            print(f"    ✓ Depths: {pipeline_depths.shape if pipeline_depths is not None else 'None'}")
            
            if pipeline_depths is not None:
                d_flat = pipeline_depths.reshape(-1)
                print(f"      Depth stats (GPU): min={d_flat.min():.4f}, max={d_flat.max():.4f}, mean={d_flat.mean():.4f}")
        
        # DepthSplat GPU fallback: use depthsplat_results from S1, compute raw_gaussians
        if args.model == 'depthsplat' and depthsplat_results is not None:
            with torch.no_grad():
                b_ds, v_ds = context['image'].shape[:2]
                pipeline_depths = depthsplat_results['depth_preds'][-1]
                match_prob = depthsplat_results['match_probs'][-1]
                match_prob_max = torch.max(match_prob, dim=1, keepdim=True)[0]
                if match_prob_max.shape[-2:] != pipeline_depths.shape[-2:]:
                    match_prob_max = F.interpolate(match_prob_max, size=pipeline_depths.shape[-2:], mode='nearest')
                pipeline_densities = rearrange(
                    match_prob_max,
                    "(b v) c h w -> b v (c h w) () ()", b=b_ds, v=v_ds
                )
                
                # Compute raw_gaussians via decomposed encoder sub-modules
                features_upsampled = model.encoder.feature_upsampler(
                    depthsplat_results["features_mono_intermediate"],
                    cnn_features=depthsplat_results["features_cnn_all_scales"][::-1],
                    mv_features=depthsplat_results["features_mv"][0] if model.encoder.cfg.num_scales == 1
                               else depthsplat_results["features_mv"][::-1]
                )
                concat_input = torch.cat((
                    rearrange(context["image"], "b v c h w -> (b v) c h w"),
                    rearrange(pipeline_depths, "b v h w -> (b v) () h w"),
                    match_prob_max,
                    features_upsampled,
                ), dim=1)
                regressor_out = model.encoder.gaussian_regressor(concat_input)
                gaussian_head_input = torch.cat([
                    regressor_out,
                    rearrange(context["image"], "b v c h w -> (b v) c h w"),
                    features_upsampled,
                    match_prob_max,
                ], dim=1)
                pipeline_raw_gaussians = model.encoder.gaussian_head(gaussian_head_input)
                
            print(f"    ✓ Depths (from S1): {pipeline_depths.shape}")
            print(f"    ✓ raw_gaussians: {pipeline_raw_gaussians.shape}")
            d_flat = pipeline_depths.reshape(-1)
            print(f"      Depth stats: min={d_flat.min():.4f}, max={d_flat.max():.4f}, mean={d_flat.mean():.4f}")

    fsdr_feature_source = 'pipeline'
    fsdr_features = pipeline_features
    fsdr_feature_dim = CONFIG.feature_dim
    if pipeline_features is not None:
        from scripts.fsdr_trace import (
            runtime_fsdr_feature_dim,
            select_fsdr_feature_tensor,
        )

        fsdr_features, fsdr_feature_source = select_fsdr_feature_tensor(
            model=args.model,
            source=args.fsdr_feature_source,
            pipeline_features=pipeline_features,
            mono_features=pipeline_mono_features,
        )
        fsdr_feature_dim = runtime_fsdr_feature_dim(fsdr_features)
    fsdr = FSDRSimulator(
        feature_dim=fsdr_feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold,
        reuse_hamming=CONFIG.fsdr_reuse_hamming,
        reuse_spatial=CONFIG.fsdr_reuse_spatial,
        reuse_confidence=CONFIG.fsdr_reuse_confidence,
        num_depth_candidates=CONFIG.num_depth_candidates,
        seed=CONFIG.fsdr_seed,
        guidance_policy=args.fsdr_guidance_policy,
        depth_consistency_threshold=CONFIG.gamma_depth,
        tile_size=CONFIG.tile_size,
    )
    print(
        f"    FSDR feature source: {fsdr_feature_source} "
        f"({fsdr_feature_dim} channels)"
    )

    if args.fsdr_only:
        if (
            fsdr_features is None
            or fsdr_depth_probs is None
            or fsdr_depth_candidates is None
            or fsdr_candidate_domain != 'inverse_depth'
            or not fsdr_probability_source
        ):
            raise RuntimeError(
                "FSDR-only claim requires authentic full-search probabilities and candidates"
            )
        from scripts.fsdr_trace import (
            prepare_fsdr_candidate_frames,
            tile_probe_pixel_order,
        )

        fsdr_frames = prepare_fsdr_candidate_frames(
            fsdr_features,
            fsdr_depth_probs,
            fsdr_depth_candidates,
        )
        fsdr_shape = fsdr_frames[0][4]
        fsdr_height, fsdr_width = fsdr_shape
        pixel_count_per_frame = fsdr_height * fsdr_width
        for frame_index, (
            feature_frame,
            candidate_anchors,
            top1_indices,
            candidate_frame,
            frame_shape,
        ) in enumerate(fsdr_frames):
            if frame_shape != fsdr_shape:
                raise RuntimeError("FSDR context frames have inconsistent evidence shapes")
            fsdr.begin_frame()
            fsdr.process_discrete_frame(
                feature_frame,
                candidate_anchors,
                top1_indices,
                candidate_frame,
                width=fsdr_width,
                pixel_order=tile_probe_pixel_order(
                    height=fsdr_height,
                    width=fsdr_width,
                    tile_size=CONFIG.tile_size,
                ),
                pixel_index_offset=frame_index * pixel_count_per_frame,
            )
        summary = fsdr.get_summary()
        if summary.get('discrete_candidate_evidence') is not True:
            raise RuntimeError("FSDR discrete candidate evidence is incomplete")
        from scripts.fsdr_evidence import (
            build_fsdr_sample_record,
            validate_fsdr_record,
            write_json,
        )
        from scripts.result_record import (
            build_environment_provenance,
            cached_sha256_file,
            portable_command,
            sha256_file,
            source_identity,
        )

        source = source_identity()
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'fsdr-demo'
        )
        protocol_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        record = build_fsdr_sample_record(
            provenance={
                'git_commit': source['git_commit'],
                'git_dirty': source['git_dirty'],
                'source_identity': source['source'],
                'source_tree_sha256': source['source_tree_sha256'],
                'submodules': source['submodules'],
                'model': args.model,
                'dataset': {
                    'name': args.dataset,
                    'representation': dataset_identity['representation'],
                    'tree_sha256': dataset_identity['tree_sha256'],
                    'manifest': str(dataset_manifest.relative_to(SCARF_ROOT)),
                    'manifest_sha256': sha256_file(dataset_manifest),
                    'paper_result_eligible': args.claim_run,
                },
                'checkpoint': {
                    'path': str(checkpoint_path.relative_to(SCARF_ROOT)),
                    'sha256': cached_sha256_file(checkpoint_path),
                    'load': getattr(model, '_scarf_checkpoint_load', {}),
                },
                'environment': build_environment_provenance(
                    experiment.environment_profile
                ),
                'runtime_assets': runtime_assets,
                'target_rgb': {
                    'loaded_by_native_dataloader': native_dataloader_loaded_target_rgb,
                    'removed_before_device_transfer_or_execution': (
                        target_rgb_removed_before_execution
                    ),
                    'passed_to_model': False,
                    'used_for_routing_or_metric': False,
                },
                'command': portable_command(
                    [
                        sys.executable,
                        str(SCARF_ROOT / 'scripts/demo.py'),
                        *(list(argv) if argv is not None else sys.argv[1:]),
                    ]
                ),
                'seed': args.seed,
                'fsdr_feature': {
                    'source': fsdr_feature_source,
                    'dimension': fsdr_feature_dim,
                },
                'device': {
                    'type': device.type,
                    'name': (
                        torch.cuda.get_device_name(device)
                        if device.type == 'cuda'
                        else str(device)
                    ),
                },
                'evaluation': {
                    'kind': 'sample',
                    'sample_index': protocol_index,
                    'execution_index': args.sample_index,
                    'candidate_count': args.num_samples,
                    'scene': str(scene_name),
                    'context_indices': [
                        int(value)
                        for value in batch['context']['index'][0].tolist()
                    ],
                    'target_indices': [
                        int(value)
                        for value in batch['target']['index'][0, :V_tgt].tolist()
                    ],
                },
            },
            kind=(
                'fsdr_sample'
                if args.claim_run
                else 'fsdr_target_free_audit'
            ),
            total_pixels=int(summary['total_pixels']),
            cache_hits=int(summary['cache_hits']),
            cache_misses=int(summary['cache_misses']),
            guided_pixels=int(summary['guided']),
            guided_in_window=int(summary['guided_in_window']),
            guided_out_window=int(summary['guided_out_window']),
            guided_top1_covered=int(summary['guided_top1_covered']),
            guided_top1_missed=int(summary['guided_top1_missed']),
            depth_inconsistent=int(summary.get('depth_inconsistent', 0)),
            hit_no_guide=int(summary['hit_no_guide']),
            full_depth_candidates=CONFIG.num_depth_candidates,
            narrowed_depth_candidates=CONFIG.fsdr_narrowed_candidates,
            candidate_domain=fsdr_candidate_domain,
            probability_source=fsdr_probability_source,
            evidence_height=fsdr_height,
            evidence_width=fsdr_width,
        )
        validate_fsdr_record(record)
        write_json(record, output_dir / 'results.json')
        print(
            "  FSDR-only evidence: "
            f"guided={record['fsdr']['metrics']['guided_rate']*100:.2f}%, "
            f"top1={record['fsdr']['metrics']['top1_coverage']*100:.4f}%"
        )
        print(f"Structured result: {output_dir / 'results.json'}")
        return

    if args.saes_routing_audit:
        if pipeline_features is None or pipeline_depths is None:
            raise RuntimeError(
                "SAES routing audit requires executed S1 features and S2 depths"
            )
        from scripts.saes_diagnostics import decision_statistics
        from scripts.result_record import (
            cached_sha256_file,
            portable_command,
            sha256_file,
            source_identity,
            write_result,
        )

        audit_record = decision_statistics(
            pipeline_features,
            pipeline_depths,
            height=h,
            width=w,
            tile_size=CONFIG.tile_size,
            feature_threshold=CONFIG.feature_var_threshold,
            depth_threshold=CONFIG.depth_std_threshold,
            near=near,
            far=far,
        )
        source = source_identity()
        protocol_sample_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        audit_record.update(
            {
                'model': args.model,
                'dataset': args.dataset,
                'dataset_representation': dataset_identity['representation'],
                'dataset_tree_sha256': dataset_identity['tree_sha256'],
                'dataset_manifest': str(dataset_manifest.relative_to(SCARF_ROOT)),
                'dataset_manifest_sha256': sha256_file(dataset_manifest),
                'checkpoint': {
                    'path': str(checkpoint_path.relative_to(SCARF_ROOT)),
                    'sha256': cached_sha256_file(checkpoint_path),
                    'load': getattr(model, '_scarf_checkpoint_load', {}),
                },
                'source': source,
                'command': portable_command(
                    [
                        sys.executable,
                        str(SCARF_ROOT / 'scripts/demo.py'),
                        *(list(argv) if argv is not None else sys.argv[1:]),
                    ]
                ),
                'seed': args.seed,
                'sample_index': args.sample_index,
                'protocol_sample_index': protocol_sample_index,
                'scene': str(scene_name),
                'context_indices': [
                    int(value) for value in batch['context']['index'][0].tolist()
                ],
                'target_indices': [
                    int(value)
                    for value in batch['target']['index'][0, :V_tgt].tolist()
                ],
                'target_rgb_provenance': {
                    'loaded_by_native_dataloader': native_dataloader_loaded_target_rgb,
                    'removed_before_device_transfer_or_execution': (
                        target_rgb_removed_before_execution
                    ),
                    'passed_to_model': False,
                    'used_for_routing_or_metric': False,
                },
                'execution_boundary': {
                    'completed': ('S1', 'S2', 'S3_raw_descriptor_head'),
                    'baseline_encoder_executed': False,
                    'ggu_s4_executed': False,
                    'renderer_executed': False,
                    'quality_metrics_computed': False,
                },
                'audit_configuration': {
                    'tile_size': CONFIG.tile_size,
                    'feature_threshold': CONFIG.feature_var_threshold,
                    'depth_threshold': CONFIG.depth_std_threshold,
                    'decision_semantics': args.saes_decision_semantics,
                    'feature_source': args.saes_feature_source,
                    'depth_coordinate_candidates': (
                        'metric_depth',
                        'relative_depth',
                        'normalized_inverse_depth',
                    ),
                },
                'full_s2_reference_use': 'posthoc-diagnostic-only',
                'paper_result_eligible': False,
            }
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'saes-routing-audit'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_result(audit_record, output_dir / 'results.json')
        print(
            "Target-free SAES routing-statistics audit: "
            f"{output_dir / 'results.json'}"
        )
        return

    if args.saes_s3_raw_audit:
        # This audit intentionally terminates after the model's raw Gaussian
        # descriptor head.  Do not instantiate GGU/S4 or the renderer: the
        # only complete descriptors are the post-hoc reference for measuring
        # whether selected probe descriptors can predict withheld S3 outputs.
        if (
            pipeline_features is None
            or pipeline_depths is None
            or pipeline_densities is None
            or pipeline_raw_gaussians is None
        ):
            raise RuntimeError(
                "S3 raw audit requires executed S1 features, S2 depths/densities, "
                "and raw Gaussian descriptors"
            )
        gpp = getattr(model.encoder.cfg, 'gaussians_per_pixel', 1)
        num_surfaces = getattr(model.encoder.cfg, 'num_surfaces', 1)
        if args.model == 'transplat':
            from src.geometry.projection import sample_image_grid
        elif args.model == 'mvsplat':
            MVSPLAT_ROOT = SCARF_ROOT / 'mvsplat'
            sys.path.insert(0, str(MVSPLAT_ROOT))
            from src.geometry.projection import sample_image_grid
        elif args.model == 'depthsplat':
            DEPTHSPLAT_ROOT = SCARF_ROOT / 'depthsplat'
            sys.path.insert(0, str(DEPTHSPLAT_ROOT))
            from src.geometry.projection import sample_image_grid
        else:
            raise RuntimeError(f"unsupported model for S3 raw audit: {args.model}")

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").to(device)
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        if args.model == 'depthsplat' and pipeline_raw_gaussians.dim() == 4:
            raw_by_view = rearrange(
                pipeline_raw_gaussians,
                "(b v) c h w -> b v (h w) c",
                b=B,
                v=V_ctx,
            )
            raw_without_opacity = raw_by_view[..., 1:]
            raw_parsed = rearrange(
                raw_without_opacity,
                "b v r (srf c) -> b v r srf c",
                srf=num_surfaces,
            )
            opacities_for_audit = raw_by_view[..., :1].sigmoid().unsqueeze(-1)
            raw_descriptors = raw_parsed[..., 2:]
            offset_xy = raw_parsed[..., :2].sigmoid()
            depths_for_audit = pipeline_depths
            if depths_for_audit.dim() == 4:
                depths_for_audit = rearrange(
                    depths_for_audit, "b v h w -> b v (h w) () ()"
                )
        else:
            raw_parsed = rearrange(
                pipeline_raw_gaussians,
                "b v r (srf c) -> b v r srf c",
                srf=num_surfaces,
            )
            offset_xy = raw_parsed[..., :2].sigmoid()
            raw_descriptors = raw_parsed[..., 2:]
            if hasattr(model.encoder, 'map_pdf_to_opacity'):
                opacities_for_audit = (
                    model.encoder.map_pdf_to_opacity(pipeline_densities, 0) / gpp
                )
            else:
                opacities_for_audit = pipeline_densities.sigmoid() / gpp
            depths_for_audit = pipeline_depths
        coordinates_for_audit = xy_ray + (offset_xy - 0.5) * pixel_size

        from scripts.saes_diagnostics import raw_s3_probe_interpolation_audit
        from scripts.result_record import (
            cached_sha256_file,
            portable_command,
            sha256_file,
            source_identity,
            write_result,
        )

        audit_record = raw_s3_probe_interpolation_audit(
            raw_descriptors,
            depths_for_audit,
            opacities_for_audit,
            coordinates_for_audit,
            features=pipeline_features,
            height=h,
            width=w,
            tile_size=CONFIG.tile_size,
            feature_threshold=CONFIG.feature_var_threshold,
            depth_threshold=CONFIG.depth_std_threshold,
            decision_semantics=args.saes_decision_semantics,
        )
        source = source_identity()
        protocol_sample_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        audit_record.update(
            {
                'model': args.model,
                'dataset': args.dataset,
                'dataset_representation': dataset_identity['representation'],
                'dataset_tree_sha256': dataset_identity['tree_sha256'],
                'dataset_manifest': str(dataset_manifest.relative_to(SCARF_ROOT)),
                'dataset_manifest_sha256': sha256_file(dataset_manifest),
                'checkpoint': {
                    'path': str(checkpoint_path.relative_to(SCARF_ROOT)),
                    'sha256': cached_sha256_file(checkpoint_path),
                    'load': getattr(model, '_scarf_checkpoint_load', {}),
                },
                'source': source,
                'command': portable_command(
                    [
                        sys.executable,
                        str(SCARF_ROOT / 'scripts/demo.py'),
                        *(list(argv) if argv is not None else sys.argv[1:]),
                    ]
                ),
                'seed': args.seed,
                'sample_index': args.sample_index,
                'protocol_sample_index': protocol_sample_index,
                'scene': str(scene_name),
                'context_indices': [
                    int(value) for value in batch['context']['index'][0].tolist()
                ],
                'target_indices': [
                    int(value)
                    for value in batch['target']['index'][0, :V_tgt].tolist()
                ],
                'target_rgb_provenance': {
                    'loaded_by_native_dataloader': native_dataloader_loaded_target_rgb,
                    'removed_before_device_transfer_or_execution': (
                        target_rgb_removed_before_execution
                    ),
                    'passed_to_model': False,
                    'used_for_routing_or_metric': False,
                },
                'execution_boundary': {
                    'completed': ('S1', 'S2', 'S3_raw_descriptor_head'),
                    'ggu_s4_executed': False,
                    'baseline_encoder_executed': False,
                    'renderer_executed': False,
                    'quality_metrics_computed': False,
                },
                'audit_configuration': {
                    'tile_size': CONFIG.tile_size,
                    'feature_threshold': CONFIG.feature_var_threshold,
                    'depth_threshold': CONFIG.depth_std_threshold,
                    'decision_semantics': args.saes_decision_semantics,
                    'feature_source': args.saes_feature_source,
                },
                'paper_result_eligible': False,
            }
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'saes-s3-raw-audit'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_result(audit_record, output_dir / 'results.json')
        print(
            "Target-free S3-before-S4 raw-descriptor audit: "
            f"{output_dir / 'results.json'}"
        )
        return

    # ============================================================
    # STAGE 3: Gaussian Generation
    # ============================================================
    print("  [Stage 3] Gaussian Generation...")
    # Keep a valid descriptor layout for accounting even in a non-strict
    # all-hardware-disabled debugging run.  Model adapters overwrite these
    # defaults below whenever a native Gaussian configuration is available.
    actual_sh_degree = CONFIG.sh_degree
    actual_scale_min = CONFIG.scale_min
    actual_scale_max = CONFIG.scale_max
    
    # Special case: if ALL hardware simulators are disabled, use baseline directly
    # This avoids error accumulation from running components separately
    all_hw_disabled = args.no_feature and args.no_depth and args.no_gaussian
    if all_hw_disabled:
        print(f"    Mode: Baseline (all HW disabled, using encoder output directly)")
        scarf_gaussians_full = baseline_gaussians
        
        # Skip the rest of Stage 3 logic
        use_gaussian_sim = False
        has_gaussian_inputs = False
    else:
        # Get model config for gaussian generation
        gpp = model.encoder.cfg.gaussians_per_pixel if hasattr(model.encoder.cfg, 'gaussians_per_pixel') else 1
        num_surfaces = model.encoder.cfg.num_surfaces if hasattr(model.encoder.cfg, 'num_surfaces') else 1
        
        # Get GGU config from model
        if hasattr(model.encoder, 'gaussian_adapter') and hasattr(model.encoder.gaussian_adapter, 'cfg'):
            ga_cfg = model.encoder.gaussian_adapter.cfg
            actual_sh_degree = ga_cfg.sh_degree
            actual_scale_min = ga_cfg.gaussian_scale_min
            actual_scale_max = ga_cfg.gaussian_scale_max
        else:
            actual_sh_degree = CONFIG.sh_degree
            actual_scale_min = CONFIG.scale_min
            actual_scale_max = CONFIG.scale_max
        
        use_gaussian_sim = not args.no_gaussian
        
        # Check if we have required inputs for gaussian generation
        has_gaussian_inputs = (
            pipeline_depths is not None and
            (pipeline_raw_gaussians is not None or pipeline_densities is not None)
        )
    
    if use_gaussian_sim and has_gaussian_inputs:
        # Use SCARF GGU hardware simulator
        print(f"    Mode: Hardware Simulator (GGU)")
        
        # Model-specific GGU configuration
        if args.model == 'depthsplat':
            ggu_config = GGUConfig(
                scale_min=actual_scale_min,
                scale_max=actual_scale_max,
                sh_degree=actual_sh_degree,
                image_shape=(h, w),
                scale_activation='softplus',
                use_depth_scaling=False,
                softplus_shift=-4.0,
                direction_normalize='z',
            )
        else:
            ggu_config = GGUConfig(
                scale_min=actual_scale_min,
                scale_max=actual_scale_max,
                sh_degree=actual_sh_degree,
                image_shape=(h, w),
                scale_activation='sigmoid',
                use_depth_scaling=True,
                direction_normalize='norm',
            )
        
        ggu = GGUProcessor(ggu_config, enable_cycle_counting=True)
        print(f"    GGU config: sh_degree={actual_sh_degree}, scale=[{actual_scale_min}, {actual_scale_max}]")
        
        # Import model-specific functions
        if args.model == 'transplat':
            from src.misc.sh_rotation import rotate_sh as model_rotate_sh
            from src.geometry.projection import sample_image_grid
        elif args.model == 'mvsplat':
            MVSPLAT_ROOT = SCARF_ROOT / 'mvsplat'
            sys.path.insert(0, str(MVSPLAT_ROOT))
            from src.misc.sh_rotation import rotate_sh as model_rotate_sh
            from src.geometry.projection import sample_image_grid
        elif args.model == 'depthsplat':
            DEPTHSPLAT_ROOT = SCARF_ROOT / 'depthsplat'
            sys.path.insert(0, str(DEPTHSPLAT_ROOT))
            from src.misc.sh_rotation import rotate_sh as model_rotate_sh
            from src.geometry.projection import sample_image_grid
        else:
            model_rotate_sh = None
            sample_image_grid = None
        
        try:
            # Get camera parameters
            ctx_extrinsics = context['extrinsics']
            ctx_intrinsics = context['intrinsics']
            
            # Compute xy_ray coordinates
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").to(device)
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            
            # Model-specific parsing
            if args.model == 'depthsplat' and pipeline_raw_gaussians is not None and pipeline_raw_gaussians.dim() == 4:
                # DepthSplat: parse gaussian_head_output [BV, C, H, W]
                B_ds, V_ds = B, V_ctx
                raw_g_bv = rearrange(pipeline_raw_gaussians, "(b v) c h w -> b v (h w) c", b=B_ds, v=V_ds)
                opacities = raw_g_bv[..., :1].sigmoid().unsqueeze(-1)
                raw_g_full = raw_g_bv[..., 1:]
                raw_gaussians_parsed = rearrange(raw_g_full, "b v r (srf c) -> b v r srf c", srf=num_surfaces)
                offset_xy = raw_gaussians_parsed[..., :2].sigmoid()
                xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                raw_gaussians_for_ggu = raw_gaussians_parsed[..., 2:]
                depths_for_ggu = pipeline_depths
                if depths_for_ggu.dim() == 4:
                    depths_for_ggu = rearrange(depths_for_ggu, "b v h w -> b v (h w) () ()")
            else:
                # TranSplat/MVSplat
                raw_gaussians_parsed = rearrange(pipeline_raw_gaussians, "b v r (srf c) -> b v r srf c", srf=num_surfaces)
                offset_xy = raw_gaussians_parsed[..., :2].sigmoid()
                xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                if hasattr(model.encoder, 'map_pdf_to_opacity'):
                    opacities = model.encoder.map_pdf_to_opacity(pipeline_densities, 0) / gpp
                else:
                    opacities = pipeline_densities.sigmoid() / gpp
                raw_gaussians_for_ggu = raw_gaussians_parsed[..., 2:]
                depths_for_ggu = pipeline_depths
            
            sh_input_images = context['image'] if args.model == 'depthsplat' else None
            
            # Call GGU
            ggu_means, ggu_covs, ggu_harmonics, ggu_opacities = ggu.forward_batch(
                ctx_extrinsics,
                ctx_intrinsics,
                xy_ray,
                depths_for_ggu,
                opacities,
                raw_gaussians_for_ggu,
                (h, w),
                rotate_sh_func=model_rotate_sh,
                input_images=sh_input_images,
            )
            
            # Reshape to [B, N, ...]
            ggu_means = rearrange(ggu_means, "b v r srf gpp xyz -> b (v r srf gpp) xyz")
            ggu_covs = rearrange(ggu_covs, "b v r srf gpp i j -> b (v r srf gpp) i j")
            ggu_harmonics = rearrange(ggu_harmonics, "b v r srf gpp c sh -> b (v r srf gpp) c sh")
            ggu_opacities = rearrange(ggu_opacities, "b v r srf gpp -> b (v r srf gpp)")
            
            from src.model.types import Gaussians
            scarf_gaussians_full = Gaussians(
                means=ggu_means,
                covariances=ggu_covs,
                harmonics=ggu_harmonics,
                opacities=ggu_opacities,
            )
            
            # The hardware-only diagnostic intentionally has no baseline
            # encoder or target-image path.  Its event ledger remains valid
            # without this numerical comparison.
            if args.saes_hardware_audit:
                print("    GGU baseline comparison: skipped (target-free hardware audit)")
            else:
                mse_means = F.mse_loss(ggu_means, baseline_gaussians.means)
                ggu_psnr_means = -10 * torch.log10(mse_means + 1e-10).item()
                print(f"    ✓ GGU vs Baseline PSNR (means): {ggu_psnr_means:.2f} dB")
            
        except Exception as e:
            if strict_run:
                raise strict_stage_error("GGU simulator", e) from e
            print(f"    ⚠ GGU error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_gaussian_sim = False
    
    if not all_hw_disabled and (not use_gaussian_sim or not has_gaussian_inputs):
        if strict_run:
            raise RuntimeError(
                "strict run GGU has incomplete inputs; GPU fallback is forbidden"
            )
        fallback_stages.append("gaussian")
        # Use original GPU for gaussian generation
        print(f"    Mode: Original GPU" + (" (fallback)" if use_gaussian_sim else " (--no-gaussian)"))
        
        if has_gaussian_inputs and hasattr(model.encoder, 'gaussian_adapter'):
            # Import necessary functions
            if args.model == 'transplat':
                from src.misc.sh_rotation import rotate_sh as model_rotate_sh
                from src.geometry.projection import sample_image_grid
            elif args.model == 'mvsplat':
                MVSPLAT_ROOT = SCARF_ROOT / 'mvsplat'
                sys.path.insert(0, str(MVSPLAT_ROOT))
                from src.misc.sh_rotation import rotate_sh as model_rotate_sh
                from src.geometry.projection import sample_image_grid
            elif args.model == 'depthsplat':
                DEPTHSPLAT_ROOT = SCARF_ROOT / 'depthsplat'
                sys.path.insert(0, str(DEPTHSPLAT_ROOT))
                from src.misc.sh_rotation import rotate_sh as model_rotate_sh
                from src.geometry.projection import sample_image_grid
            else:
                model_rotate_sh = None
                sample_image_grid = None
            
            if sample_image_grid is not None:
                pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
                xy_ray, _ = sample_image_grid((h, w), device)
                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").to(device)
                
                # Parse raw_gaussians - model-specific format
                if args.model == 'depthsplat' and pipeline_raw_gaussians is not None and pipeline_raw_gaussians.dim() == 4:
                    # DepthSplat: parse gaussian_head_output [BV, C, H, W]
                    # First convert to [B, V, H*W, C] format
                    B_ds, V_ds = B, V_ctx
                    raw_g_bv = rearrange(pipeline_raw_gaussians, "(b v) c h w -> b v (h w) c", b=B_ds, v=V_ds)
                    # Then parse as usual
                    raw_gaussians_parsed = rearrange(raw_g_bv, "b v r (srf c) -> b v r srf c", srf=num_surfaces)
                    
                    # DepthSplat format: opacity(1) + offset_xy(2) + scales(3) + rotations(4) + sh(27) = 37
                    opacities_raw = raw_gaussians_parsed[..., :1].sigmoid()  # [B, V, H*W, srf, 1]
                    opacities = opacities_raw / gpp  # [B, V, H*W, srf, 1] - gpp=1
                    offset_xy = raw_gaussians_parsed[..., 1:3].sigmoid()
                    raw_gaussians_for_ga = raw_gaussians_parsed[..., 3:]  # scales + rotations + sh = 34
                else:
                    # TranSplat/MVSplat format: [B, V, H*W, C]
                    raw_gaussians_parsed = rearrange(pipeline_raw_gaussians, "b v r (srf c) -> b v r srf c", srf=num_surfaces)
                    
                    if args.model == 'depthsplat':
                        # DepthSplat format but in [B, V, H*W, C] format (unlikely but handle it)
                        opacities_raw = raw_gaussians_parsed[..., :1].sigmoid()
                        opacities = opacities_raw / gpp
                        offset_xy = raw_gaussians_parsed[..., 1:3].sigmoid()
                        raw_gaussians_for_ga = raw_gaussians_parsed[..., 3:]
                    else:
                        # TranSplat/MVSplat format: offset_xy(2) + scales(3) + rotations(4) + sh(75) = 84
                        offset_xy = raw_gaussians_parsed[..., :2].sigmoid()
                        raw_gaussians_for_ga = raw_gaussians_parsed[..., 2:]
                        # Compute opacities from densities
                        if hasattr(model.encoder, 'map_pdf_to_opacity'):
                            opacities = model.encoder.map_pdf_to_opacity(pipeline_densities, 0) / gpp
                        else:
                            opacities = pipeline_densities.sigmoid() / gpp
                
                xy_ray_adjusted = xy_ray + (offset_xy - 0.5) * pixel_size
                
                # Prepare inputs for GaussianAdapter
                ctx_extrinsics = rearrange(context['extrinsics'], "b v i j -> b v () () () i j")
                ctx_intrinsics = rearrange(context['intrinsics'], "b v i j -> b v () () () i j")
                xy_ray_for_adapter = rearrange(xy_ray_adjusted, "b v r srf xy -> b v r srf () xy")
                raw_gaussians_for_adapter = rearrange(
                    raw_gaussians_for_ga,
                    "b v r srf c -> b v r srf () c",
                )
                
                # Use original GaussianAdapter - model-specific call
                if args.model == 'depthsplat':
                    # DepthSplat requires input_images for SH initialization
                    ga_output = model.encoder.gaussian_adapter.forward(
                        ctx_extrinsics,
                        ctx_intrinsics,
                        xy_ray_for_adapter,
                        pipeline_depths,  # Input from Stage 2
                        opacities,
                        raw_gaussians_for_adapter,
                        (h, w),
                        input_images=context['image'],
                    )
                else:
                    # TranSplat/MVSplat - no input_images
                    ga_output = model.encoder.gaussian_adapter.forward(
                        ctx_extrinsics,
                        ctx_intrinsics,
                        xy_ray_for_adapter,
                        pipeline_depths,  # Input from Stage 2
                        opacities,
                        raw_gaussians_for_adapter,
                        (h, w),
                    )
                
                # Reshape to [B, N, ...]
                from src.model.types import Gaussians
                ga_means = rearrange(ga_output.means, "b v r srf gpp xyz -> b (v r srf gpp) xyz")
                ga_covs = rearrange(ga_output.covariances, "b v r srf gpp i j -> b (v r srf gpp) i j")
                ga_harmonics = rearrange(ga_output.harmonics, "b v r srf gpp c sh -> b (v r srf gpp) c sh")
                ga_opacities = rearrange(ga_output.opacities, "b v r srf gpp -> b (v r srf gpp)")
                
                scarf_gaussians_full = Gaussians(
                    means=ga_means,
                    covariances=ga_covs,
                    harmonics=ga_harmonics,
                    opacities=ga_opacities,
                )
                
                # Verify against baseline
                mse_means = F.mse_loss(scarf_gaussians_full.means, baseline_gaussians.means)
                adapter_psnr = -10 * torch.log10(mse_means + 1e-10).item()
                print(f"    ✓ GaussianAdapter vs Baseline PSNR: {adapter_psnr:.2f} dB")
            else:
                # Fallback to baseline
                print(f"    ⚠ No gaussian generation available, using baseline")
                scarf_gaussians_full = baseline_gaussians
        else:
            # No inputs or no gaussian_adapter - use baseline
            print(f"    ⚠ Using baseline gaussians (no pipeline inputs)")
            scarf_gaussians_full = baseline_gaussians
    
    if args.saes_feature_source == 'pipeline':
        saes_features = pipeline_features
    else:
        if pipeline_saes_features is None:
            raise RuntimeError(
                "Gaussian-head SAES feature source was requested but not captured"
            )
        if pipeline_saes_features.dim() != 5:
            raise RuntimeError("Gaussian-head SAES features are not [B,V,C,H,W]")
        if tuple(pipeline_saes_features.shape[:2]) != (B, V_ctx):
            raise RuntimeError("Gaussian-head SAES features do not cover every view")
        if tuple(pipeline_saes_features.shape[-2:]) != (h, w):
            raise RuntimeError("Gaussian-head SAES features are not full resolution")
        saes_features = pipeline_saes_features
    print(
        f"    SAES feature source: {args.saes_feature_source} "
        f"({saes_features.shape[2]} channels, "
        f"{saes_features.shape[-2]}x{saes_features.shape[-1]})"
    )
    depths = pipeline_depths
    densities = pipeline_densities
    raw_gaussians_captured = pipeline_raw_gaussians
    
    # --------------------------------------------------------
    # Threshold Tuning (optional: --tune-thresholds)
    # --------------------------------------------------------
    if args.tune_thresholds:
        print()
        print("=" * 70)
        print("THRESHOLD TUNING MODE")
        print("=" * 70)
        gt_image_tune = target['image'][0, 0]
        baseline_psnr_tune = -10 * torch.log10(F.mse_loss(baseline_image, gt_image_tune)).item()
        
        best_saes_th, best_fsdr_tol, _, _ = tune_thresholds(
            scarf_gaussians_full, model, tgt_ext, tgt_int, target, h, w, device,
            gt_image_tune, baseline_psnr_tune,
            features=saes_features, depths=pipeline_depths,
        )
        
        # Apply tuned thresholds (already written to CONFIG inside tune_thresholds)
        print(f"\n  Applying tuned thresholds: feat_var={CONFIG.feature_var_threshold}, "
              f"depth_std={CONFIG.depth_std_threshold}")
        
        # Re-create FSDR with tuned config
        fsdr = FSDRSimulator(
            feature_dim=fsdr_feature_dim,
            cache_size=CONFIG.fsdr_cache_size,
            hamming_threshold=CONFIG.fsdr_hamming_threshold,
            reuse_hamming=CONFIG.fsdr_reuse_hamming,
            reuse_spatial=CONFIG.fsdr_reuse_spatial,
            reuse_confidence=CONFIG.fsdr_reuse_confidence,
            num_depth_candidates=CONFIG.num_depth_candidates,
            seed=CONFIG.fsdr_seed,
            guidance_policy=args.fsdr_guidance_policy,
            depth_consistency_threshold=CONFIG.gamma_depth,
            tile_size=CONFIG.tile_size,
        )
        # Reset savings tracker
        savings = SavingsTracker(model_type=args.model)
        print("=" * 70)
    
    # --------------------------------------------------------
    # Step 4b-4d: Real Ablation (SAES v4 L0+L1 + FSDR + 4-config render)
    # --------------------------------------------------------
    from src.model.types import Gaussians
    N = scarf_gaussians_full.means.shape[1]
    saes_low_var_stats = {
        'early_tiles': 0,
        'low_var_tiles': 0,
        'low_var_agree': 0.0,
        'mean_similarity': 0.0,
        'threshold': 0.90,
    }
    saes_diagnostic_record = None
    saes_representatives = None
    saes_diagnostic_images = {}
    
    # Clone original Gaussians BEFORE any modifications (needed for ablation configs)
    orig_means = scarf_gaussians_full.means.clone()
    orig_covs = scarf_gaussians_full.covariances.clone()
    orig_harmo = scarf_gaussians_full.harmonics.clone()
    orig_opacs = scarf_gaussians_full.opacities.clone()

    if args.calibration_trace:
        from scripts.calibration_replay import replay_calibration_sample

        if batch.get('calibration', {}).get('target_rgb_accessed') is not False:
            raise RuntimeError("calibration input provenance does not prove target isolation")

        def _calibration_render(gaussians_obj):
            with torch.no_grad():
                rendered = model.decoder.forward(
                    gaussians_obj,
                    tgt_ext,
                    tgt_int,
                    target['near'],
                    target['far'],
                    (h, w),
                    depth_mode=None,
                )
            return rendered.color[0, :V_tgt]

        candidates, trace = replay_calibration_sample(
            config=CONFIG,
            gaussians=scarf_gaussians_full,
            features=saes_features,
            depths=depths,
            render=_calibration_render,
            gaussian_type=Gaussians,
            image_height=h,
            image_width=w,
            context_extrinsics=context['extrinsics'],
            context_intrinsics=context['intrinsics'],
            ray_depth_mode=('z' if args.model == 'depthsplat' else 'euclidean'),
            committed_parameters=args.calibration_parameters,
        )
        trace['calibration_input'] = batch['calibration']
        protocol_sample_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        record = {
            'schema_version': '1.0',
            'kind': 'calibration_sample_trace',
            'sample_index': protocol_sample_index,
            'execution_index': args.sample_index,
            'scene': str(scene_name),
            'context_indices': [
                int(value) for value in batch['context']['index'][0].tolist()
            ],
            'target_indices': [
                int(value) for value in batch['target']['index'][0, :V_tgt].tolist()
            ],
            'model': args.model,
            'dataset': args.dataset,
            'seed': args.seed,
            'trace': trace,
            'calibration_scope': (
                'exact_committed_train_tuple'
                if args.calibration_parameters is not None
                else 'registered_global_grid'
            ),
            'committed_parameters': args.calibration_parameters,
            'candidates': candidates,
        }
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'calibration-trace'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        from scripts.result_record import write_result

        write_result(record, output_dir / 'results.json')
        print(f"Calibration trace: {output_dir / 'results.json'}")
        return

    if args.sensitivity_trace:
        from scripts.sensitivity_replay import replay_sample
        from scripts.sensitivity_sweep import STUDIES

        def _sensitivity_render(gaussians_obj):
            with torch.no_grad():
                rendered = model.decoder.forward(
                    gaussians_obj,
                    tgt_ext,
                    tgt_int,
                    target['near'],
                    target['far'],
                    (h, w),
                    depth_mode=None,
                )
            return rendered.color[0, :V_tgt]

        replays, trace = replay_sample(
            studies=STUDIES,
            config=CONFIG,
            gaussians=scarf_gaussians_full,
            features=saes_features,
            depths=depths,
            target_images=target['image'][0, :V_tgt],
            render=_sensitivity_render,
            gaussian_type=Gaussians,
            savings_tracker_type=SavingsTracker,
            cycle_counter_type=HWCycleCounter,
            image_height=h,
            image_width=w,
            feature_cycles=feature_sim_cycles,
            depth_cycles=depth_sim_cycles,
            dp_core_cycles=dp_core_cycles,
            cost_volume_cycles=cost_volume_cycles,
            gauss_gen_cycles=gauss_gen_cycles,
            s1_cnn_cycles=s1_cnn_cycles,
            context_extrinsics=context['extrinsics'],
            context_intrinsics=context['intrinsics'],
            ray_depth_mode=('z' if args.model == 'depthsplat' else 'euclidean'),
        )
        protocol_sample_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        selection = {
            'sample_index': protocol_sample_index,
            'scene': str(scene_name),
            'context_indices': [
                int(value) for value in batch['context']['index'][0].tolist()
            ],
            'target_indices': [
                int(value)
                for value in batch['target']['index'][0, :V_tgt].tolist()
            ],
        }
        from scripts.result_record import cached_sha256_file, source_identity

        source = source_identity()
        calibration_provenance = {
            key: value
            for key, value in mechanism_provenance.items()
            if key != 'mechanism_config_sha256'
        }

        trace_record = {
            'schema_version': '1.0',
            'kind': 'sensitivity_sample_trace',
            **selection,
            'execution_index': args.sample_index,
            'candidate_count': args.num_samples,
            'model': args.model,
            'dataset': args.dataset,
            'dataset_representation': dataset_identity['representation'],
            'dataset_tree_sha256': dataset_identity['tree_sha256'],
            'checkpoint_sha256': cached_sha256_file(checkpoint_path),
            'seed': args.seed,
            'execution_provenance': {
                'git_commit': source['git_commit'],
                'git_dirty': source['git_dirty'],
                'source_identity': source['source'],
                'source_tree_sha256': source['source_tree_sha256'],
                'submodules': source['submodules'],
                'mechanism_config_sha256': mechanism_provenance[
                    'mechanism_config_sha256'
                ],
                'calibration_provenance': calibration_provenance,
            },
            'trace': trace,
            'replays': replays,
        }
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'sensitivity-trace'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        from scripts.result_record import write_result

        write_result(trace_record, output_dir / 'results.json')
        print(f"Sensitivity trace: {output_dir / 'results.json'}")
        return
    
    # ---- Step 4b: SAES v4 (L0+L1: feature + depth) ----
    if args.no_saes:
        print("  [4b] Progressive SAES v4: SKIPPED (--no-saes)")
        modified_mask = torch.zeros(N, dtype=torch.bool, device=device)
        saes_stats = {
            'total_tiles_processed': 0, 'early_stop_phase1': 0,
            'early_stop_phase2': 0, 'full_processed': 0,
            'early_stop_ratio': 0.0, 'pixels_interpolated': 0,
            'pixels_original': N, 'interpolation_ratio': 0.0,
            'level0_tiles': 0, 'level1_tiles': 0,
            'full_tiles': 0, 'level0_pixels': 0, 'level1_pixels': 0,
            'total_modified_pixels': 0,
            'level0_ratio': 0.0, 'level1_ratio': 0.0,
            'full_ratio': 1.0, 'modification_ratio': 0.0,
        }
        all_pixels = [(y, x, y * w + x) for y in range(h) for x in range(w)]
    else:
        print("  [4b] Progressive SAES v4 (L0+L1 probe moment matching)...")

        # SAES works on a clone so we preserve originals for ablation
        saes_gaussians = Gaussians(
            means=scarf_gaussians_full.means.clone(),
            covariances=scarf_gaussians_full.covariances.clone(),
            harmonics=scarf_gaussians_full.harmonics.clone(),
            opacities=scarf_gaussians_full.opacities.clone(),
        )

        modified_mask, saes_stats, continue_pixels = apply_progressive_saes(
            saes_gaussians, h, w, CONFIG.tile_size, gpp=1,
            feature_var_threshold=CONFIG.feature_var_threshold,
            depth_std_threshold=CONFIG.depth_std_threshold,
            features=saes_features,
            depths=depths,
            cross_check_threshold=CONFIG.saes_cross_check,
            view_count=V_ctx,
            materialization=args.saes_materialization,
            decision_semantics=args.saes_decision_semantics,
            beta_x=CONFIG.beta_x,
            beta_f=CONFIG.beta_f,
            beta_d=CONFIG.beta_d,
            num_depth_candidates=CONFIG.num_depth_candidates,
            context_extrinsics=context['extrinsics'],
            context_intrinsics=context['intrinsics'],
            ray_depth_mode=('z' if args.model == 'depthsplat' else 'euclidean'),
        )
        saes_stats['feature_source'] = args.saes_feature_source

        # Print feature variance distribution for threshold calibration
        if saes_features is not None:
            import numpy as np
            _tv, _ = ProgressiveSAES.classify_tiles_by_features(
                saes_features,
                h,
                w,
                CONFIG.tile_size,
                threshold=1.0,
                statistic="normalized-probe-total-variance",
            )
            _vals = sorted(_tv.values())
            _arr = np.array(_vals)
            _p = [1, 2, 5, 8, 10, 15, 20, 30, 50]
            print("    Feature-var distribution (L0 threshold calibration):")
            print("      " + "  ".join(f"P{p}={np.percentile(_arr,p):.4f}" for p in _p))

        # Realistic ASIC model: NO validate-after-compute.
        # In a real ASIC, non-probe pixels are never computed, so there's nothing
        # to validate against. The SAES classification decisions (L0/L1/L2) are
        # based on information available at decision time (S1 features, probe depths,
        # probe Gaussians). The interpolation quality depends on how conservative
        # the classification thresholds are.
        
        # All pixels for FSDR (runs on ALL pixels for its own ablation config)
        all_pixels = [(y, x, y * w + x) for y in range(h) for x in range(w)]
        
        modified_pixels = int(modified_mask.sum().item())
        total_tiles = saes_stats.get('total_tiles_processed', 0)
        l0_t = saes_stats.get('level0_tiles', 0)
        l1_t = saes_stats.get('level1_tiles', 0)
        f_t  = saes_stats.get('full_tiles', 0)
        print(f"    Level 0 (feature-uniform):  {l0_t} ({l0_t/max(1,total_tiles)*100:.1f}%)")
        print(f"    Level 1 (depth-uniform):    {l1_t} ({l1_t/max(1,total_tiles)*100:.1f}%)")
        print(f"    Full (no skip):             {f_t}  ({f_t/max(1,total_tiles)*100:.1f}%)")
        total_positions = V_ctx * h * w
        print(f"  ✓ Absorbed non-probe pixels: {modified_pixels:,}/{total_positions:,} "
              f"({modified_pixels/total_positions*100:.1f}%)")
        print(f"    (Realistic: no post-hoc validation, thresholds ensure quality)")

        saes_low_var_stats = compute_saes_low_var_agreement(
            scarf_gaussians_full,
            modified_mask,
            h,
            w,
            CONFIG.tile_size,
            view_count=V_ctx,
        )
        print(f"    Low-Var. Agree.: {saes_low_var_stats['low_var_agree']*100:.1f}% "
              f"({saes_low_var_stats['low_var_tiles']}/{saes_low_var_stats['early_tiles']} early tiles, "
              f"mean sim={saes_low_var_stats['mean_similarity']:.3f})")
        print(
            "    S2 candidate evaluations: "
            f"{saes_stats['executed_s2_evaluations']:,}/"
            f"{saes_stats['full_s2_evaluations']:,} executed"
        )

        if args.saes_materialization_audit:
            from scripts.saes_diagnostics import materialization_attribute_audit
            from scripts.result_record import write_result

            primitives_per_pixel = N // (V_ctx * h * w)
            audit_record = materialization_attribute_audit(
                Gaussians(
                    means=orig_means,
                    covariances=orig_covs,
                    harmonics=orig_harmo,
                    opacities=orig_opacs,
                ),
                saes_gaussians,
                features=saes_features,
                depths=depths,
                height=h,
                width=w,
                tile_size=CONFIG.tile_size,
                feature_threshold=CONFIG.feature_var_threshold,
                depth_threshold=CONFIG.depth_std_threshold,
                view_count=V_ctx,
                decision_semantics=args.saes_decision_semantics,
                materialization=args.saes_materialization,
            )
            audit_record.update(
                {
                    'model': args.model,
                    'dataset': args.dataset,
                    'sample_index': args.sample_index,
                    'protocol_sample_index': (
                        args.protocol_sample_index
                        if args.protocol_sample_index is not None
                        else args.sample_index
                    ),
                    'scene': str(scene_name),
                    'context_indices': [
                        int(value) for value in batch['context']['index'][0].tolist()
                    ],
                    'target_indices': [
                        int(value)
                        for value in batch['target']['index'][0, :V_tgt].tolist()
                    ],
                    'primitives_per_pixel': primitives_per_pixel,
                    'saes_stats': saes_stats,
                    'target_rgb_accessed': False,
                    'native_dataloader_loaded_target_rgb': (
                        native_dataloader_loaded_target_rgb
                    ),
                    'target_rgb_removed_before_execution': (
                        target_rgb_removed_before_execution
                    ),
                }
            )
            output_dir = (
                Path(args.output_dir)
                if args.output_dir
                else SCARF_ROOT / 'outputs' / 'saes-materialization-audit'
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            write_result(audit_record, output_dir / 'results.json')
            print(
                "Target-free SAES materialization audit: "
                f"{output_dir / 'results.json'}"
            )
            return

        if args.saes_diagnostic_sweep:
            from scripts.saes_diagnostics import (
                decision_statistics,
                representative_indices,
            )

            primitives_per_pixel = N // (V_ctx * h * w)
            saes_representatives = representative_indices(
                modified_mask,
                view_count=V_ctx,
                height=h,
                width=w,
                tile_size=CONFIG.tile_size,
                primitives_per_pixel=primitives_per_pixel,
            )
            saes_diagnostic_record = decision_statistics(
                saes_features,
                depths,
                height=h,
                width=w,
                tile_size=CONFIG.tile_size,
                feature_threshold=CONFIG.feature_var_threshold,
                depth_threshold=CONFIG.depth_std_threshold,
                near=near,
                far=far,
            )
            saes_diagnostic_record.update(
                {
                    'model': args.model,
                    'dataset': args.dataset,
                    'feature_source': args.saes_feature_source,
                    'sample_index': args.sample_index,
                    'representative_count': int(saes_representatives.numel()),
                    'coverage_sweep': [],
                    'component_attribution': [],
                    'retention_boundary': [],
                }
            )

    # Record SAES v4 savings for the cycle model.  The analytic event ledger
    # charges S3 route selection, moment matching, and retained-descriptor
    # traffic instead of treating L0/L1 bypasses as zero-cost savings.
    # Use h*w (pixel positions) as base, not N (total Gaussians incl. surfaces)
    # because ASIC processes per pixel position - skipping a position skips all surfaces.
    primitives_per_pixel = N // (V_ctx * h * w)
    saes_hardware_ledger = savings.record_saes(
        total_pixels=V_ctx * h * w,
        saes_stats=saes_stats,
        feature_dim=int(saes_features.shape[2]),
        tile_size=CONFIG.tile_size,
        sh_degree=actual_sh_degree,
        primitives_per_pixel=primitives_per_pixel,
        model_type=args.model,
    )
    saes_stats['hardware_accounting'] = saes_hardware_ledger
    saes_stats['execution_dependency'] = savings.saes_execution_dependency

    if args.saes_hardware_audit:
        from scripts.result_record import (
            cached_sha256_file,
            portable_command,
            sha256_file,
            source_identity,
            write_result,
        )

        source = source_identity()
        protocol_sample_index = (
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        )
        audit_ggu_counter = HWCycleCounter(ggu_pe_count=CONFIG.ggu_pe_count)
        audit_ggu_counter.add_ggu(N, sh_degree=actual_sh_degree)
        audit_record = {
            'kind': 'saes_hardware_accounting_audit',
            'model': args.model,
            'dataset': args.dataset,
            'dataset_representation': dataset_identity['representation'],
            'dataset_tree_sha256': dataset_identity['tree_sha256'],
            'dataset_manifest': str(dataset_manifest.relative_to(SCARF_ROOT)),
            'dataset_manifest_sha256': sha256_file(dataset_manifest),
            'checkpoint': {
                'path': str(checkpoint_path.relative_to(SCARF_ROOT)),
                'sha256': cached_sha256_file(checkpoint_path),
                'load': getattr(model, '_scarf_checkpoint_load', {}),
            },
            'source': source,
            'command': portable_command(
                [
                    sys.executable,
                    str(SCARF_ROOT / 'scripts/demo.py'),
                    *(list(argv) if argv is not None else sys.argv[1:]),
                ]
            ),
            'seed': args.seed,
            'sample_index': args.sample_index,
            'protocol_sample_index': protocol_sample_index,
            'scene': str(scene_name),
            'context_indices': [
                int(value) for value in batch['context']['index'][0].tolist()
            ],
            'target_indices': [
                int(value) for value in batch['target']['index'][0, :V_tgt].tolist()
            ],
            'target_rgb_provenance': {
                'loaded_by_native_dataloader': native_dataloader_loaded_target_rgb,
                'removed_before_device_transfer_or_execution': (
                    target_rgb_removed_before_execution
                ),
                'passed_to_model': False,
                'used_for_routing_or_metric': False,
            },
            'execution_boundary': {
                'completed': ('S1', 'S2', 'S3', 'S4'),
                'baseline_encoder_executed': False,
                'renderer_executed': False,
                'quality_metrics_computed': False,
                'expected_results_accessed': False,
            },
            # The Python simulator materializes a full descriptor tensor in
            # order to preserve reference numerics.  The ledger reads no
            # descriptor values, only the sparse path counters, and therefore
            # cannot be promoted to RTL timing evidence from this audit alone.
            'simulation_boundary': {
                'full_s3_descriptor_tensor_materialized': True,
                'ledger_input': 'runtime tile/anchor counters plus fixed dimensions only',
                'rtl_cycle_equivalent': False,
            },
            'stage_cycles': {
                's1_feature': feature_sim_cycles,
                's2_depth': dp_core_cycles,
                's3_gaussian': gauss_gen_cycles,
                's4_ggu': audit_ggu_counter.get_summary()['ggu_cycles'],
            },
            'saes_stats': saes_stats,
            'saes_hardware_accounting': saes_hardware_ledger,
            'paper_result_eligible': False,
        }
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else SCARF_ROOT / 'outputs' / 'saes-hardware-accounting-audit'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_result(audit_record, output_dir / 'results.json')
        print(
            "Target-free SAES hardware accounting audit: "
            f"{output_dir / 'results.json'}"
        )
        return
    
    # ---- Step 4c: FSDR (Feature-Similarity Gaussian Reuse) — Realistic ASIC ----
    if args.no_fsdr:
        print("  [4c] FSDR: SKIPPED (--no-fsdr)")
    else:
        print("  [4c] FSDR (Realistic ASIC: cache hit → use cached Gaussians)...")
        
        has_features = (fsdr_features is not None and
                       not isinstance(fsdr_features, str) and
                       hasattr(fsdr_features, 'shape'))
        
        if has_features and len(all_pixels) > 0:
            from scripts.fsdr_trace import (
                prepare_fsdr_candidate_frames,
                tile_probe_pixel_order,
            )

            if (
                fsdr_depth_probs is None
                or fsdr_depth_candidates is None
                or fsdr_candidate_domain != 'inverse_depth'
                or not fsdr_probability_source
            ):
                raise RuntimeError(
                    "FSDR claim execution requires authentic full-search "
                    "probabilities and inverse-depth candidates"
                )
            fsdr_frames = prepare_fsdr_candidate_frames(
                fsdr_features,
                fsdr_depth_probs,
                fsdr_depth_candidates,
            )
            frame_shape = fsdr_frames[0][4]
            fsdr_height, fsdr_width = frame_shape
            pixels_per_frame = fsdr_height * fsdr_width
            all_reuse_data = {}
            for frame_index, (
                feature_frame,
                candidate_anchors,
                top1_indices,
                candidate_frame,
                current_shape,
            ) in enumerate(fsdr_frames):
                if current_shape != frame_shape:
                    raise RuntimeError(
                        "FSDR context frames have inconsistent evidence shapes"
                    )
                fsdr.begin_frame()
                offset = frame_index * pixels_per_frame
                paths = fsdr.process_discrete_frame(
                    feature_frame,
                    candidate_anchors,
                    top1_indices,
                    candidate_frame,
                    width=fsdr_width,
                    pixel_order=tile_probe_pixel_order(
                        height=fsdr_height,
                        width=fsdr_width,
                        tile_size=CONFIG.tile_size,
                    ),
                    pixel_index_offset=offset,
                )
                all_reuse_data.update(fsdr.reuse_data)
                for pixel_idx, path in enumerate(paths):
                    savings.record_fsdr_pixel(
                        path, reused=(offset + pixel_idx in fsdr.reuse_data)
                    )
            fsdr.reuse_data = all_reuse_data
        
        fsdr_stats = fsdr.get_summary()
        if fsdr_stats['total_pixels'] > 0:
            print(f"    Pixels processed: {fsdr_stats['total_pixels']:,}")
            print(f"    Cache hit rate: {fsdr_stats['hit_rate']*100:.1f}%")
            print(f"    Guided (narrowed): {fsdr_stats.get('guided', 0):,} "
                  f"({fsdr_stats.get('guided_rate', 0)*100:.1f}%)")
            if fsdr.guidance_policy == 'historical-depth-guard':
                print(f"    Depth inconsistent: {fsdr_stats.get('depth_inconsistent', 0):,}")
                print(f"    Not guided: {fsdr_stats.get('hit_no_guide', 0):,}")
            print(f"    Full compute (miss): {fsdr_stats['full_compute']:,}")
            criteria = f"hamming≤{fsdr.config.hamming_threshold}"
            if fsdr.guidance_policy == 'historical-depth-guard':
                criteria += f", depth_consist≤{fsdr.depth_consistency_threshold:.0%}"
            print(f"    Criteria: {criteria}")
    
    # Record GGU cycles (with compact writeback savings modelled for Stage 4)
    cycle_counter.add_ggu(N, sh_degree=actual_sh_degree)
    
    # ---- Step 4d: Build 4 Ablation Gaussian Configs + Render (Realistic) ----
    print()
    print("[5/6] Real Ablation: 4-config rendering (realistic ASIC behavior)...")
    
    def _render(gaussians_obj):
        """Render a Gaussian set for every selected target view."""
        with torch.no_grad():
            out = model.decoder.forward(
                gaussians_obj, tgt_ext, tgt_int,
                target['near'], target['far'], (h, w), depth_mode=None
            )
        return out.color[0, :V_tgt]
    
    ablation_renders = {}
    
    # Config 1: No optimizations (original Gaussians)
    print("  [1/4] Rendering: no optimizations...")
    g_noopt = Gaussians(means=orig_means, covariances=orig_covs,
                        harmonics=orig_harmo, opacities=orig_opacs)
    ablation_renders['asic'] = _render(g_noopt)
    
    # Config 2: +FSDR only (narrowed search: most pixels zero quality impact)
    if not args.no_fsdr and len(fsdr.reuse_data) > 0:
        out_window_count = sum(1 for v in fsdr.reuse_data.values() if not v.get('in_window', True))
        print(f"  [2/4] Rendering: +FSDR only ({len(fsdr.reuse_data):,} guided, "
              f"{out_window_count} outside window)...")
        fsdr_means = orig_means.clone()
        for pixel_idx, reuse_info in fsdr.reuse_data.items():
            # Narrowed search: only modify means for pixels where actual depth
            # was outside the narrowed window (rare, depth consistency prevents most)
            depth_ratio = reuse_info['depth_ratio']
            if abs(depth_ratio - 1.0) > 1e-6:  # Only modify if ratio != 1.0
                fsdr_means[0, pixel_idx] = orig_means[0, pixel_idx] * depth_ratio
        g_fsdr = Gaussians(means=fsdr_means, covariances=orig_covs.clone(),
                           harmonics=orig_harmo.clone(), opacities=orig_opacs.clone())
        ablation_renders['asic_fsdr'] = _render(g_fsdr)
        depth_err = fsdr.get_depth_error_stats()
        if depth_err['count'] > 0:
            print(f"    Out-of-window depth error: mean={depth_err['mean']:.4f}, "
                  f"p95={depth_err['p95']:.4f}, max={depth_err['max']:.4f}")
        else:
            print(f"    All guided pixels had actual depth within window → zero quality impact")
    else:
        print("  [2/4] Rendering: +FSDR only (no guided / disabled)...")
        ablation_renders['asic_fsdr'] = ablation_renders['asic']
    
    # Config 3: +SAES only (SAES-interpolated appearance, no post-hoc validation)
    if not args.no_saes:
        print("  [3/4] Rendering: +SAES only (probe moment matching)...")
        ablation_renders['asic_saes'] = _render(saes_gaussians)
    else:
        ablation_renders['asic_saes'] = ablation_renders['asic']
    
    # Config 4: +FSDR + SAES (SAES interpolation + FSDR depth-only reuse on non-SAES pixels)
    if not args.no_saes or (not args.no_fsdr and len(fsdr.reuse_data) > 0):
        print("  [4/4] Rendering: +FSDR+SAES (combined realistic)...")
        # Start with SAES-modified Gaussians (or originals if SAES disabled)
        if not args.no_saes:
            combo_means = saes_gaussians.means.clone()
            combo_covs = saes_gaussians.covariances.clone()
            combo_harmo = saes_gaussians.harmonics.clone()
            combo_opacs = saes_gaussians.opacities.clone()
        else:
            combo_means = orig_means.clone()
            combo_covs = orig_covs.clone()
            combo_harmo = orig_harmo.clone()
            combo_opacs = orig_opacs.clone()
        
        # Apply FSDR narrowed search means modification on non-SAES pixels only
        if not args.no_fsdr:
            fsdr_on_non_saes = 0
            fsdr_modified = 0
            for pixel_idx, reuse_info in fsdr.reuse_data.items():
                if not modified_mask[pixel_idx]:  # Not modified by SAES
                    fsdr_on_non_saes += 1
                    depth_ratio = reuse_info['depth_ratio']
                    if abs(depth_ratio - 1.0) > 1e-6:
                        combo_means[0, pixel_idx] = orig_means[0, pixel_idx] * depth_ratio
                        fsdr_modified += 1
            print(f"    FSDR guided {fsdr_on_non_saes:,} non-SAES pixels "
                  f"({fsdr_modified} means-modified)")
        
        g_combo = Gaussians(means=combo_means, covariances=combo_covs,
                            harmonics=combo_harmo, opacities=combo_opacs)
        ablation_renders['asic_fsdr_saes'] = _render(g_combo)
    else:
        ablation_renders['asic_fsdr_saes'] = ablation_renders['asic']
    
    # Use the combined configuration as the SCARF output.
    scarf_images = ablation_renders['asic_fsdr_saes']
    scarf_time = time.time() - t0
    
    # Gaussian stats
    fsdr_reused = fsdr.stats['total_reuse'] if not args.no_fsdr else 0
    gaussian_stats = {
        'gaussians_baseline': N,
        'gaussians_output': saes_stats.get('effective_gaussians', N),
        'pixels_modified': saes_stats.get('total_modified_pixels', 0),
        'fsdr_reused': fsdr_reused,
    }
    
    print(f"  ✓ 4-config rendering complete ({scarf_time:.2f}s total)")
    print(f"  ✓ SAES absorbed {gaussian_stats['pixels_modified']:,} non-probe pixels")
    print(f"  ✓ FSDR guided {fsdr_reused:,} pixels (narrowed search, {CONFIG.fsdr_narrowed_candidates}/{CONFIG.num_depth_candidates} candidates)")
    
    # --------------------------------------------------------
    # Step 6: Evaluate per-config quality and Save
    # --------------------------------------------------------
    print()
    print("[6/6] Evaluating per-config quality...")
    from scripts.result_record import mean_view_quality
    
    gt_images = target['image'][0, :V_tgt]
    
    # Metrics
    def compute_psnr(img1, img2):
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        mse = (img1 - img2).square().flatten(1).mean(dim=1)
        return (-10 * torch.log10(mse)).mean().item()
    
    def compute_ssim(img1, img2):
        from torchmetrics.functional.image import structural_similarity_index_measure

        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        return structural_similarity_index_measure(
            img1, img2, data_range=1.0
        ).item()

    lpips_metric = get_lpips_metric(gt_images.device)

    def compute_lpips(img1, img2):
        with torch.no_grad():
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
            value = lpips_metric(img1, img2, normalize=True)
        return value.reshape(-1).mean().item()

    def compute_view_metrics(images, references):
        return [
            {
                'psnr_db': compute_psnr(images[index], references[index]),
                'ssim': compute_ssim(images[index], references[index]),
                'lpips': compute_lpips(images[index], references[index]),
            }
            for index in range(images.shape[0])
        ]
    
    # Per-config quality (real ablation)
    # Reference: SCARF no-opt is the "correct" ASIC output.
    # Quality loss is measured relative to SCARF no-opt, NOT GPU baseline.
    ablation_quality = {}
    
    ablation_view_metrics = {
        cfg_key: compute_view_metrics(cfg_images, gt_images)
        for cfg_key, cfg_images in ablation_renders.items()
    }
    noopt_views = ablation_view_metrics['asic']
    noopt_psnr = sum(item['psnr_db'] for item in noopt_views) / len(noopt_views)
    noopt_ssim = sum(item['ssim'] for item in noopt_views) / len(noopt_views)
    
    for cfg_key, cfg_image in ablation_renders.items():
        view_metrics = ablation_view_metrics[cfg_key]
        psnr = sum(item['psnr_db'] for item in view_metrics) / len(view_metrics)
        ssim = sum(item['ssim'] for item in view_metrics) / len(view_metrics)
        lpips = sum(item['lpips'] for item in view_metrics) / len(view_metrics)
        # Loss vs SCARF no-opt (the "correct" ASIC output)
        loss_db = psnr - noopt_psnr
        loss_pct = abs(loss_db) / noopt_psnr * 100 if noopt_psnr > 0 else 0
        ablation_quality[cfg_key] = {
            'psnr': psnr, 'ssim': ssim, 'lpips': lpips,
            'loss_db': loss_db, 'loss_pct': loss_pct,
            'quality_views': [
                {
                    'target_index': int(
                        batch['target']['index'][0, view_index].item()
                    ),
                    **metrics,
                }
                for view_index, metrics in enumerate(view_metrics)
            ],
        }
        ref_label = "(reference)" if cfg_key == 'asic' else f"loss={loss_db:+.4f} dB ({loss_pct:.4f}%)"
        print(f"  {cfg_key:<20s}: PSNR={psnr:.4f} dB, SSIM={ssim:.6f}, {ref_label}")
    
    # Use FSDR+SAES as the "SCARF" result
    baseline_view_metrics = compute_view_metrics(baseline_images, gt_images)
    scarf_view_metrics = compute_view_metrics(scarf_images, gt_images)
    quality_views = [
        {
            'target_index': int(batch['target']['index'][0, view_index].item()),
            'baseline': baseline_view_metrics[view_index],
            'scarf': scarf_view_metrics[view_index],
        }
        for view_index in range(V_tgt)
    ]
    baseline_quality = mean_view_quality(quality_views, 'baseline')
    scarf_quality = mean_view_quality(quality_views, 'scarf')
    baseline_psnr = baseline_quality['psnr_db']
    baseline_ssim = baseline_quality['ssim']
    baseline_lpips = baseline_quality['lpips']
    scarf_psnr = scarf_quality['psnr_db']
    scarf_ssim = scarf_quality['ssim']
    scarf_lpips = scarf_quality['lpips']

    if args.saes_diagnostic_sweep:
        from scripts.saes_diagnostics import (
            build_component_variant,
            build_coverage_variant,
            build_ranked_tile_subset_variant,
            probe_cross_check_errors,
            probe_feature_variances,
        )

        if saes_diagnostic_record is None or saes_representatives is None:
            raise RuntimeError("SAES diagnostic state was not initialized")

        def record_diagnostic(
            destination,
            *,
            key,
            diagnostic_kind,
            diagnostic_images,
            fields,
        ):
            view_metrics = compute_view_metrics(diagnostic_images, gt_images)
            quality = {
                metric: float(
                    sum(view[metric] for view in view_metrics) / len(view_metrics)
                )
                for metric in ('psnr_db', 'ssim', 'lpips')
            }
            delta = {
                metric: quality[metric] - baseline_quality[metric]
                for metric in ('psnr_db', 'ssim', 'lpips')
            }
            accepted = (
                abs(delta['psnr_db']) <= 0.15
                and abs(delta['ssim']) <= 0.005
                and abs(delta['lpips']) <= 0.005
            )
            destination.append(
                {
                    'key': key,
                    'diagnostic_kind': diagnostic_kind,
                    **fields,
                    'quality': quality,
                    'delta_from_baseline': delta,
                    'within_existing_tolerances': accepted,
                }
            )
            print(
                f"    {key}: dPSNR={delta['psnr_db']:+.4f} dB, "
                f"dSSIM={delta['ssim']:+.6f}, "
                f"dLPIPS={delta['lpips']:+.6f}, pass={accepted}"
            )
            saes_diagnostic_images[key] = diagnostic_images.detach().cpu()

        # A projected 2D Gaussian's integral scales approximately with its
        # covariance multiplier and alpha. Pair the two in opposite directions
        # so this sweep tests coverage rather than simply adding opacity mass.
        mass_conserving_pairs = (
            (0.5, 2.0),
            (1.0, 1.0),
            (1.5, 2.0 / 3.0),
            (2.0, 0.5),
            (3.0, 1.0 / 3.0),
            (4.0, 0.25),
        )
        print("  Diagnostic mass-conserving coverage sweep:")
        for covariance_scale, alpha_exponent in mass_conserving_pairs:
            key = f"mass_cov_{covariance_scale:g}_alpha_{alpha_exponent:.6g}"
            if covariance_scale == 1.0 and alpha_exponent == 1.0:
                diagnostic_images = ablation_renders['asic_saes']
            else:
                variant = build_coverage_variant(
                    saes_gaussians,
                    saes_representatives,
                    covariance_scale=covariance_scale,
                    opacity_scale=alpha_exponent,
                )
                diagnostic_images = _render(variant)
            record_diagnostic(
                saes_diagnostic_record['coverage_sweep'],
                key=key,
                diagnostic_kind='mass_conserving_coverage',
                diagnostic_images=diagnostic_images,
                fields={
                    'covariance_scale': covariance_scale,
                    'alpha_exponent': alpha_exponent,
                },
            )

        component_variants = (
            ('restore_means', ('means',)),
            ('restore_covariances', ('covariances',)),
            ('restore_harmonics', ('harmonics',)),
            ('restore_opacities', ('opacities',)),
            ('restore_means_covariances', ('means', 'covariances')),
            ('restore_harmonics_opacities', ('harmonics', 'opacities')),
            (
                'zero_only',
                ('means', 'covariances', 'harmonics', 'opacities'),
            ),
        )
        print("  Diagnostic representative-component attribution:")
        for key, restored_components in component_variants:
            variant = build_component_variant(
                saes_gaussians,
                g_noopt,
                saes_representatives,
                restore=restored_components,
            )
            diagnostic_images = _render(variant)
            record_diagnostic(
                saes_diagnostic_record['component_attribution'],
                key=key,
                diagnostic_kind='representative_component_attribution',
                diagnostic_images=diagnostic_images,
                fields={'restored_components': list(restored_components)},
            )

        tile_scores = probe_feature_variances(
            saes_features,
            height=h,
            width=w,
            tile_size=CONFIG.tile_size,
        )
        current_rate = float(saes_stats['level0_ratio'])
        retention_fractions = sorted(
            {
                current_rate * numerator / 4.0
                for numerator in range(1, 5)
            }
        )
        print("  Diagnostic target-free L0 retention boundary:")
        for target_fraction in retention_fractions:
            variant, metadata = build_ranked_tile_subset_variant(
                g_noopt,
                saes_gaussians,
                modified_mask,
                tile_scores,
                view_count=V_ctx,
                height=h,
                width=w,
                tile_size=CONFIG.tile_size,
                primitives_per_pixel=primitives_per_pixel,
                target_fraction=target_fraction,
                ranking_statistic='raw_probe_vector_variance',
            )
            diagnostic_images = _render(variant)
            key = f"retention_{metadata['selected_fraction']:.6f}"
            record_diagnostic(
                saes_diagnostic_record['retention_boundary'],
                key=key,
                diagnostic_kind='target_free_retention_boundary',
                diagnostic_images=diagnostic_images,
                fields={
                    **metadata,
                    'boundary_source': 'executed_current_level0_rate',
                },
            )

        cross_check_scores = probe_cross_check_errors(
            g_noopt,
            view_count=V_ctx,
            height=h,
            width=w,
            tile_size=CONFIG.tile_size,
            primitives_per_pixel=primitives_per_pixel,
        )
        print("  Diagnostic target-free probe-error retention boundary:")
        for target_fraction in retention_fractions:
            variant, metadata = build_ranked_tile_subset_variant(
                g_noopt,
                saes_gaussians,
                modified_mask,
                cross_check_scores,
                view_count=V_ctx,
                height=h,
                width=w,
                tile_size=CONFIG.tile_size,
                primitives_per_pixel=primitives_per_pixel,
                target_fraction=target_fraction,
                ranking_statistic='probe_gaussian_leave_one_out_error',
            )
            diagnostic_images = _render(variant)
            key = f"probe_error_retention_{metadata['selected_fraction']:.6f}"
            record_diagnostic(
                saes_diagnostic_record['retention_boundary'],
                key=key,
                diagnostic_kind='target_free_probe_error_retention_boundary',
                diagnostic_images=diagnostic_images,
                fields={
                    **metadata,
                    'boundary_source': 'executed_current_level0_rate',
                },
            )
    
    # Check quality budget (compare optimized configs vs no-opt, excluding no-opt itself)
    opt_configs = {k: v for k, v in ablation_quality.items() if k != 'asic'}
    worst_loss = max(q['loss_pct'] for q in opt_configs.values()) if opt_configs else 0
    if worst_loss <= 1.0:
        print(f"  ✓ All optimized configs within 1.0% quality budget (worst: {worst_loss:.4f}%)")
    else:
        print(f"  ⚠ Quality budget exceeded: worst config {worst_loss:.4f}% > 1.0%")
    
    # Save outputs
    output_dir = Path(args.output_dir) if args.output_dir else SCARF_ROOT / 'outputs' / 'demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    if saes_diagnostic_record is not None:
        saes_diagnostic_record['selection'] = {
            'scene': str(scene_name),
            'context_indices': [
                int(value) for value in batch['context']['index'][0].tolist()
            ],
            'target_indices': [
                int(value)
                for value in batch['target']['index'][0, :V_tgt].tolist()
            ],
        }
        saes_diagnostic_record['command'] = [
            sys.executable,
            str(SCARF_ROOT / 'scripts/demo.py'),
            *(list(argv) if argv is not None else sys.argv[1:]),
        ]
        with open(output_dir / 'saes_diagnostic.json', 'w', encoding='utf-8') as stream:
            json.dump(saes_diagnostic_record, stream, indent=2, sort_keys=True)
            stream.write('\n')
    
    save_images = args.image_output_policy == 'all' or (
        args.image_output_policy == 'representative' and args.sample_index == 0
    )
    if save_images:
        from torchvision.utils import save_image
        for view_index in range(V_tgt):
            save_image(gt_images[view_index], output_dir / f'gt_{view_index:02d}.png')
            save_image(
                baseline_images[view_index],
                output_dir / f'baseline_{view_index:02d}.png',
            )
            save_image(
                scarf_images[view_index], output_dir / f'scarf_{view_index:02d}.png'
            )
            for cfg_key, cfg_images in ablation_renders.items():
                save_image(
                    cfg_images[view_index],
                    output_dir / f'ablation_{cfg_key}_{view_index:02d}.png',
                )
            for key, diagnostic_images in saes_diagnostic_images.items():
                save_image(
                    diagnostic_images[view_index],
                    output_dir / f'saes_{key}_{view_index:02d}.png',
                )
    
    # Get statistics
    ggu_stats = cycle_counter.get_summary()
    fsdr_summary = fsdr.get_summary()
    preservation_metrics = {
        'model': args.model,
        'sample_index': args.sample_index,
        'image_size': [h, w],
        'fsdr_top1_cov': fsdr_summary.get('in_window_rate', 0.0),
        'fsdr_guided_pixels': fsdr_summary.get('guided', 0),
        'fsdr_guided_in_window': fsdr_summary.get('guided_in_window', 0),
        'fsdr_guided_out_window': fsdr_summary.get('guided_out_window', 0),
        'saes_low_var_agree': saes_low_var_stats.get('low_var_agree', 0.0),
        'saes_low_var_tiles': saes_low_var_stats.get('low_var_tiles', 0),
        'saes_early_tiles': saes_low_var_stats.get('early_tiles', 0),
        'saes_low_var_mean_similarity': saes_low_var_stats.get('mean_similarity', 0.0),
        'saes_low_var_threshold': saes_low_var_stats.get('threshold', 0.90),
    }
    with open(output_dir / 'preservation_metrics.json', 'w') as f:
        json.dump(preservation_metrics, f, indent=2)
    
    # Base cycle counts (no SAES/FSDR savings)
    base_ggu_cycles = ggu_stats['ggu_cycles']
    
    from scripts.result_record import require_positive_cycles

    require_positive_cycles({
        'feature': feature_sim_cycles,
        'depth': dp_core_cycles,
        'gaussian': gauss_gen_cycles,
        'ggu': base_ggu_cycles,
    })
    
    # Compute ablation table (with real savings from SAES/FSDR)
    # FSDR narrowing ratio: 1 - narrowed/original
    fsdr_narrow_ratio = 1.0 - (CONFIG.fsdr_narrowed_candidates / CONFIG.num_depth_candidates
                                if CONFIG.num_depth_candidates > 0 else 0.25)
    
    ablation = savings.compute_ablation(
        feature_cycles=feature_sim_cycles,
        depth_cycles=depth_sim_cycles,
        ggu_cycles=base_ggu_cycles,
        dp_core_cycles=dp_core_cycles,
        gauss_gen_cycles=gauss_gen_cycles,
        cost_volume_cycles=cost_volume_cycles,
        s1_cnn_cycles=s1_cnn_cycles,
        ablation_quality=ablation_quality,
        fsdr_narrowing_ratio=fsdr_narrow_ratio,
    )
    pipe_info = ablation.get('_pipeline', {})
    saes_stats['s2_s3_saving'] = {
        's2': pipe_info.get('saes_s2_saving', 0.0),
        's3': pipe_info.get('saes_s3_saving', 0.0),
    }
    
    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS - SCARF Demo (Real Ablation)")
    print("=" * 70)
    print()
    print("### Per-Config Quality Metrics (Real Ablation)")
    print(f"  BASELINE (GPU):        PSNR={baseline_psnr:.4f} dB, SSIM={baseline_ssim:.6f}")
    cfg_labels = {
        'asic': 'ASIC (no opt)',
        'asic_fsdr': 'ASIC + FSDR',
        'asic_saes': 'ASIC + SAES v4',
        'asic_fsdr_saes': 'ASIC + SAES+FSDR',
    }
    for cfg_key in ['asic', 'asic_fsdr', 'asic_saes', 'asic_fsdr_saes']:
        q = ablation_quality.get(cfg_key, {})
        label = cfg_labels.get(cfg_key, cfg_key)
        print(f"  {label:<24s}: PSNR={q.get('psnr', 0):.4f} dB, SSIM={q.get('ssim', 0):.6f}, "
              f"loss={q.get('loss_db', 0):+.4f} dB ({q.get('loss_pct', 0):.4f}%)")
    print()
    print("### SAES v4 Results (Multi-Level, Dataflow-Aligned)")
    print(f"  Total tiles:    {saes_stats.get('total_tiles_processed', 0)}")
    print(f"  Level 0 (feat): {saes_stats.get('level0_tiles', 0)} tiles "
          f"({saes_stats.get('level0_ratio', 0)*100:.1f}%), "
          f"{savings.level0_pixels:,} non-probe px absorbed (opacity→0)")
    print(f"  Level 1 (depth):{saes_stats.get('level1_tiles', 0)} tiles "
          f"({saes_stats.get('level1_ratio', 0)*100:.1f}%), "
          f"{savings.level1_pixels:,} non-probe px absorbed (opacity→0)")
    print(f"  Full (no skip): {saes_stats.get('full_tiles', 0)} tiles "
          f"({saes_stats.get('full_ratio', 0)*100:.1f}%)")
    print(f"  Total modified: {savings.saes_interpolated_pixels:,}/{savings.total_pixels:,} "
          f"({savings.saes_interpolation_ratio()*100:.1f}%)")
    _eff = saes_stats.get('effective_gaussians', savings.total_pixels)
    _zer = saes_stats.get('zeroed_gaussians', 0)
    _tot = saes_stats.get('total_tiles_processed', 1) * (CONFIG.tile_size ** 2)
    print(f"  Effective Gaussians (opacity>0): {_eff:,}  "
          f"Zeroed (opacity=0): {_zer:,}  "
          f"({_zer / max(1, _tot) * 100:.1f}% of tile px suppressed)")
    print()
    print("### FSDR Statistics (Narrowed Depth Search ASIC Model)")
    fsdr_summary = fsdr.get_summary()
    if fsdr_summary['total_pixels'] > 0:
        print(f"  Pixels processed:     {fsdr_summary['total_pixels']:,}")
        print(f"  Cache hit rate:       {fsdr_summary['hit_rate']*100:.1f}%")
        print(f"  Guided (32 cand.):    {fsdr_summary.get('guided', 0):,} "
              f"({fsdr_summary['guided_rate']*100:.1f}%)")
        print(f"    In window:          {fsdr_summary.get('guided_in_window', 0):,} "
              f"({fsdr_summary.get('in_window_rate', 0)*100:.1f}% → zero quality impact)")
        print(f"    Out of window:      {fsdr_summary.get('guided_out_window', 0):,} "
              f"(means modified)")
        if fsdr.guidance_policy == 'historical-depth-guard':
            print(f"  Depth inconsistent:   {fsdr_summary.get('depth_inconsistent', 0):,} "
                  f"(→ full 128-candidate search)")
            print(f"  Not guided (criteria):{fsdr_summary.get('hit_no_guide', 0):,} "
                  f"({fsdr_summary.get('hit_no_guide_rate', 0)*100:.1f}%)")
        print(f"  Full compute (miss):  {fsdr_summary['full_compute']:,} "
              f"({fsdr_summary['full_compute_rate']*100:.1f}%)")
        fsdr_save_pct = pipe_info.get('fsdr_s2_save_per_pixel', 0.42) * 100
        print(f"  S2 saving per guided: {fsdr_save_pct:.1f}% "
              f"(cost_vol+UNet+DepthHead, 32/{CONFIG.num_depth_candidates} candidates)")
        depth_err = fsdr_summary.get('depth_error', {})
        if depth_err.get('count', 0) > 0:
            print(f"  Out-window depth err: mean={depth_err['mean']:.4f}, max={depth_err['max']:.4f}")
        criteria = f"hamming≤{fsdr.config.hamming_threshold}"
        if fsdr.guidance_policy == 'historical-depth-guard':
            criteria += f", depth_consist≤{fsdr.depth_consistency_threshold:.0%}"
        print(f"  Criteria:             {criteria}")
    else:
        print(f"  (No FSDR pixels processed)")
    
    # --------------------------------------------------------
    # Hardware Configuration
    # --------------------------------------------------------
    print()
    print("### SCARF Hardware Configuration (High-Performance)")
    from encoder.types import GEMMConfig, BilinearConfig
    _gemm_cfg = GEMMConfig()
    _bilinear_cfg = BilinearConfig()
    _dp_vw = 32  # default vector width for depth predictor
    print(f"  Clock:            {SCARF_FREQ_MHZ} MHz")
    from encoder.types import ConvConfig as _ConvCfg
    _conv_cfg = _ConvCfg()
    # Show upgraded hardware config
    hw_sc = CONFIG.hw_scale_compute
    hw_sm = CONFIG.hw_scale_memory
    print(f"  HW Scale (compute): {hw_sc:.1f}x (48²/32²=2.25x MACs, -11% overhead)")
    print(f"  HW Scale (memory):  {hw_sm:.1f}x (SRAM BW ∝ array row width 48/32)")
    print(f"  ConvEngine:       48×48 = 2304 MACs (logical)")
    print(f"  GEMM Unit:        48×48 = 2304 MACs (logical)")
    print(f"  Total MACs:       4608 MACs @ {SCARF_FREQ_MHZ} MHz = {4608*SCARF_FREQ_MHZ/1000:.1f} GOPS")
    print(f"  Vector ALU:       64-wide SIMD")
    print(f"  BilinearUnit:     64 ch × 32 samplers")
    print(f"  GGU PEs:          {CONFIG.ggu_pe_count}")
    print(f"  (Base simulators: Conv {_conv_cfg.pe_array_size}×{_conv_cfg.pe_array_size}, "
          f"GEMM {_gemm_cfg.tile_m}×{_gemm_cfg.tile_n})")
    
    # Print pipeline model parameters
    print()
    print("### ASIC Pipeline Model")
    print(f"  HW Scale (compute): {pipe_info.get('HW_SCALE_C', 2.0):.1f}x  "
          f"(conv, GEMM, depth_head, regression, GGU)")
    print(f"  HW Scale (memory):  {pipe_info.get('HW_SCALE_M', 1.5):.1f}x  "
          f"(cost_volume bilinear warping, feature reads)")
    s1_cnn_b = pipe_info.get('S1_CNN_BOOST', 1.0)
    s1_vit_b = pipe_info.get('S1_VIT_BOOST', 1.0)
    print(f"  S1 CNN boost:       {s1_cnn_b:.1f}x  "
          f"(Winograd F(2,3) + Conv-BN-ReLU fusion + weight-stationary)")
    print(f"  S1 ViT boost:       {s1_vit_b:.2f}x  "
          f"(large GEMM efficiency + LayerNorm+GELU fusion)")
    print(f"  S1 eff. CNN scale:  {pipe_info.get('HW_SCALE_C', 2.0) * s1_cnn_b:.1f}x  "
          f"(HW_SCALE×CNN_BOOST = {pipe_info.get('HW_SCALE_C', 2.0):.1f}×{s1_cnn_b:.1f})")
    print(f"  S1 eff. ViT scale:  {pipe_info.get('HW_SCALE_C', 2.0) * s1_vit_b:.2f}x  "
          f"(HW_SCALE×VIT_BOOST = {pipe_info.get('HW_SCALE_C', 2.0):.1f}×{s1_vit_b:.2f})")
    cv_frac = pipe_info.get('cv_frac_scaled', 0)
    print(f"  S2 cost_volume frac:{cv_frac*100:5.1f}%  (after scaling; memory-bound portion)")
    print(f"  S1 FE overlap:      x{pipe_info['PIPE_FE']:.2f}  "
          f"(ConvEngine || GEMM in ViT, double buffering)")
    print(f"  S2 DP overlap:      x{pipe_info['PIPE_DP']:.2f}  "
          f"(BilinearUnit || VectorALU || ConvEngine tile pipeline)")
    print(f"  S3 GaussNN overlap: x{pipe_info['PIPE_GG_NN']:.2f}  "
          f"(refine_unet → to_gauss output buf || weight prefetch)")
    print(f"  GGU Post:           hidden (dedicated PEs overlap with ConvEngine)")
    _ggu_s = ggu_stats.get('compact_writeback_bytes_saved', 0)
    _ggu_cs = ggu_stats.get('compact_writeback_cycles_saved', 0)
    print(f"  S4 compact writeback: cov=6 (upper-tri), SH={actual_sh_degree}°={(actual_sh_degree+1)**2} coeffs×3ch")
    print(f"    Bytes saved: {_ggu_s:,}  Cycles saved: {_ggu_cs:,} "
          f"({_ggu_cs / max(1, ggu_stats.get('ggu_raw_cycles', 1)) * 100:.1f}% of raw GGU cycles)")
    print(f"  SAES v4 multi-level (K(T) adaptive probes):")
    print(f"    Combined S2 save: {pipe_info.get('saes_s2_saving', 0)*100:.1f}%")
    print(f"    Combined S3 save: {pipe_info.get('saes_s3_saving', 0)*100:.1f}%")
    print(
        "    SAES control+merge: "
        f"{pipe_info.get('saes_control_accounting_cycles', 0):,} cycles "
        f"({pipe_info.get('saes_control_timing_class', 'unavailable')})"
    )
    fsdr_save_pp = pipe_info.get('fsdr_s2_save_per_pixel', 0.42)
    print(f"  FSDR guided:        {pipe_info.get('fsdr_reuse_ratio', 0)*100:.1f}% "
          f"(narrowed S2: {CONFIG.fsdr_narrowed_candidates}/{CONFIG.num_depth_candidates} candidates, "
          f"saves {fsdr_save_pp*100:.1f}%/pixel, cost_vol+UNet+DepthHead)")
    print(f"  Total S2 saving:    {pipe_info.get('combined_s2_saving', 0)*100:.1f}%")
    print(f"  Total S3 saving:    {pipe_info.get('combined_s3_saving', 0)*100:.1f}%")
    
    from scripts.result_record import (
        build_environment_provenance,
        build_result_record,
        write_result,
    )
    from scripts.validate_result import validate

    best_cycles = ablation['asic_fsdr_saes'].get(
        'eff_total', ablation['asic_fsdr_saes']['total']
    )
    # Normalize measured baseline latency to the SCARF clock so the ratio is a
    # latency speedup. Raw device timing and clock remain in provenance.
    baseline_equivalent_cycles = round(
        baseline_gpu_time_ms * SCARF_FREQ_MHZ * 1000
    )
    ablation_record = {
        key: {**value, 'quality': ablation_quality[key]}
        for key, value in ablation.items()
        if not key.startswith('_')
    }
    component_cycles = {
        'feature': feature_sim_cycles,
        'depth': dp_core_cycles,
        'gaussian': gauss_gen_cycles,
        'ggu': base_ggu_cycles,
    }
    record = build_result_record(
        model=args.model,
        dataset=args.dataset,
        checkpoint=checkpoint_path,
        checkpoint_load=getattr(model, "_scarf_checkpoint_load", {}),
        environment=build_environment_provenance(experiment.environment_profile),
        dataset_manifest=dataset_manifest,
        dataset_representation=dataset_identity['representation'],
        dataset_tree_sha256=dataset_identity['tree_sha256'],
        device={
            'type': device.type,
            'name': gpu_name if device.type == 'cuda' else str(device),
            'measured_encoder_time_ms': baseline_gpu_time_ms,
            'encoder_timing_samples_ms': baseline_timing_samples_ms,
            'encoder_timing_repetitions': len(baseline_timing_samples_ms),
            'encoder_timing_source': (
                'cuda_events' if device.type == 'cuda' else 'perf_counter'
            ),
            'reported_clock_mhz': gpu_freq_mhz,
            'scarf_clock_mhz': SCARF_FREQ_MHZ,
        },
        seed=args.seed,
        quality={
            'baseline': {
                'psnr_db': baseline_psnr,
                'ssim': baseline_ssim,
                'lpips': baseline_lpips,
            },
            'scarf': {
                'psnr_db': scarf_psnr,
                'ssim': scarf_ssim,
                'lpips': scarf_lpips,
            },
        },
        quality_views=quality_views,
        baseline_cycles=baseline_equivalent_cycles,
        cycles=component_cycles,
        scarf_cycles=best_cycles,
        cycle_source='scarf_component_simulators',
        ablation=ablation_record,
        fsdr_saes={
            'fsdr': fsdr_summary,
            'saes': saes_stats,
            'preservation': preservation_metrics,
        },
        command=[
            sys.executable,
            str(SCARF_ROOT / 'scripts/demo.py'),
            *(list(argv) if argv is not None else sys.argv[1:]),
        ],
        runtime_assets=runtime_assets,
        sample_identity={
            'scene': str(scene_name),
            'context_indices': [int(value) for value in batch['context']['index'][0].tolist()],
            'target_indices': [
                int(value) for value in batch['target']['index'][0, :V_tgt].tolist()
            ],
        },
        sample_index=(
            args.protocol_sample_index
            if args.protocol_sample_index is not None
            else args.sample_index
        ),
        execution_index=args.sample_index,
        num_samples=args.num_samples,
        baseline_source=(
            'orin_nx_cuda_events'
            if device.type == 'cuda' and 'Orin' in gpu_name
            else (
                'workstation_cuda_events'
                if device.type == 'cuda'
                else 'cpu_perf_counter'
            )
        ),
        fallback_stages=fallback_stages,
        stage_records=mmcu_stage_records(component_cycles),
        paper_result_eligible=(
            args.claim_run
            and not (
                args.saes_materialization != 'representative'
                or
                args.saes_decision_semantics != 'current'
                or args.saes_feature_source != 'pipeline'
                or args.fsdr_guidance_policy != 'paper-hamming-local-validity'
                or saes_stats.get('camera_aware_moment_matching') is not True
            )
        ),
    )
    if strict_run:
        validate(record)
    result_path = output_dir / 'results.json'
    write_result(record, result_path)
    print()
    print(f"Structured result: {result_path}")
    print("Physical PPA is intentionally excluded; run hardware/iflow/run.sh.")


if __name__ == '__main__':
    main()
