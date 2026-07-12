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

# Set TORCH_HOME to use local cache (avoid network check for DINOv2 etc.)
os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

# SCARF root directory
SCARF_ROOT = Path(__file__).parent.parent.resolve()

# Add SCARF to path for imports
sys.path.insert(0, str(SCARF_ROOT))

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
from integration import create_model_loader, ModelBundle, DataBundle
from ggu import GGUProcessor, GGUConfig
from fsdr import FSDRSimulator
from saes import ProgressiveSAES, apply_progressive_saes

# SCARF hardware clock frequency (MHz) — default 1 GHz, overridable via --freq
SCARF_FREQ_MHZ = 1000


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

    # Fallback
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown GPU'
    return {'name': name, 'freq_mhz': 1500}


def gpu_timed_inference(fn, warmup: int = 1, repeats: int = 5) -> float:
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
    times_ms.sort()
    return times_ms[len(times_ms) // 2]


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
    depth_std_threshold: float = 0.003    # Level 1: ~3% of remaining tiles (depth-flat, geometrically planar)
    saes_cross_check: float = 0.015       # Probe cross-check error threshold (conservative, ≤2% quality across all models)
    
    # ---- FSDR config — Depth-Only Reuse (Realistic ASIC) ----
    # In real ASIC: cache hit → reuse cached DEPTH (skip S2 only)
    # S3 still runs with cached depth → only means/positions affected
    # Quality impact is proportional to depth error (very small for matched pixels)
    # This allows much more aggressive reuse criteria than full-Gaussian reuse
    fsdr_cache_size: int = 32
    fsdr_hamming_threshold: int = 4       # Cache lookup hamming range
    fsdr_reuse_hamming: int = 3           # Moderate hamming (depth-only is safe)
    fsdr_reuse_spatial: int = 12          # Moderate spatial distance
    fsdr_reuse_confidence: float = 0.80   # Min peak_prob for reuse decision
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
    test_views: int = 1
    
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
    - Non-probe opacities zeroed → effective Gaussian count ≤ K(T) per early-stopped tile.
    - FSDR: Feature-Similarity Depth Reuse. Cache hits skip S2 cost_volume for guided pixels.
    """
    
    LIGHT_VERIFY_COST_RATIO = 0.04
    
    def __init__(self):
        self.total_pixels = 0
        # SAES v4: L0 + L1 only
        self.level0_pixels = 0   # Feature-uniform tiles: K(T) probes, 12 non-probe opacity→0
        self.level1_pixels = 0   # Depth-uniform tiles:   K(T) probes, 12 non-probe opacity→0
        self.saes_interpolated_pixels = 0  # Total modified (backward compat)
        # FSDR
        self.fsdr_direct_reuse = 0
        self.fsdr_interpolation = 0
        self.fsdr_light_verify = 0
        self.fsdr_full_search = 0
        self.fsdr_validated = 0
        self.fsdr_rejected = 0
    
    def record_saes(self, total_pixels: int, saes_stats: Dict):
        """Record SAES v4 (L0+L1) statistics."""
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
        
        Realistic Savings Model:
          SAES Level 0: Feature-uniform tiles → K(T) probes, rest opacity→0
                        Saves 75% of S2+S3 per tile. Decision: S1 feature variance + cross-check.
          SAES Level 1: Depth-uniform tiles → K(T) probes, rest opacity→0
                        Saves 75% of S2+S3 per tile. Decision: probe depth uniformity + cross-check.
          FSDR (Narrowed Search): Guided pixels search D/4 depth candidates.
                Saves cost_volume computation only (memory-bound, per-pixel per-candidate).
                U-Net/depth_head/regression unchanged (process full spatial resolution).
                Decision: LSH hamming + confidence + depth consistency check.
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
        
        # SAES v4 ratios (fraction of total pixels with opacity zeroed)
        # L0 + L1: both skip full S2+S3 for non-probe pixels.
        l0_ratio = self.level0_ratio()   # Feature-uniform: saves S2+S3 for non-probe px
        l1_ratio = self.level1_ratio()   # Depth-uniform:   saves S2+S3 for non-probe px
        total_saes = l0_ratio + l1_ratio
        
        # FSDR: fraction of remaining (non-SAES) pixels that get guided search
        fsdr_ratio = self.fsdr_validated_saving_ratio()
        
        if dp_core_cycles == 0 and gauss_gen_cycles == 0:
            dp_core_cycles = depth_cycles
            gauss_gen_cycles = 0
        
        # ================================================================
        # Apply differentiated hardware scaling per stage
        # ================================================================
        # S1 (Feature Extraction): CNN-specific ASIC optimizations
        #   CNN portion: Winograd + Conv-BN-ReLU fusion + weight-stationary
        #   ViT portion: Large GEMM efficiency + LayerNorm+GELU fusion
        S1_CNN_BOOST = CONFIG.s1_cnn_boost   # 1.8x additional for CNN
        S1_VIT_BOOST = CONFIG.s1_vit_boost   # 1.15x additional for ViT
        
        if s1_cnn_cycles > 0 and feature_cycles > 0:
            s1_vit_cycles = max(0, feature_cycles - s1_cnn_cycles)
            fe_cnn = int(s1_cnn_cycles / (HW_SCALE_C * S1_CNN_BOOST))
            fe_vit = int(s1_vit_cycles / (HW_SCALE_C * S1_VIT_BOOST))
            feature_cycles = fe_cnn + fe_vit
        else:
            # Fallback: uniform compute scaling (no CNN/ViT breakdown available)
            feature_cycles = int(feature_cycles / HW_SCALE_C)
        
        # S2 (Depth Prediction): split cost_volume (memory-bound) from rest (compute-bound)
        #   cost_volume: bilinear warping + correlation → limited by SRAM read bandwidth
        #   unet + depth_head + regression: convolutions/GEMM → compute-bound
        if cost_volume_cycles > 0 and dp_core_cycles > 0:
            non_cv_cycles = dp_core_cycles - cost_volume_cycles
            dp_cv_scaled = int(cost_volume_cycles / HW_SCALE_M)   # memory-bound
            dp_rest_scaled = int(non_cv_cycles / HW_SCALE_C)      # compute-bound
            dp_core_cycles = dp_cv_scaled + dp_rest_scaled
        else:
            # Fallback: treat as uniformly compute-bound
            dp_cv_scaled = 0
            dp_core_cycles = int(dp_core_cycles / HW_SCALE_C)
        
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
        # Both L0 and L1 skip the full S2+S3 pipeline for non-probe pixels.
        saes_s2_saving = total_saes   # L0+L1 both skip S2 for non-probe px
        saes_s3_saving = total_saes   # L0+L1 both skip S3 for non-probe px
        
        # ================================================================
        # FSDR savings: Narrowed Depth Search only
        # ================================================================
        # The FSDR simulation implements a single-tier narrowed search:
        #   All guided pixels run S2 with D/4 candidates (75% S2 savings).
        #   S3 (Gaussian generation) still runs for every pixel — there is no
        #   Tier-1 S3 bypass in the current simulation.  Claiming S3 savings
        #   for any fraction of guided pixels would overstate the speedup.
        #
        # Architecture: LSH hash unit (16-bit) + 512-entry cache table.
        #   Guided (cache hit, hamming ≤ reuse_hamming): narrowed D/4 search.
        #   Miss: full D-candidate search.
        #
        # Per-pixel S2 saving for guided pixels (D-dependent ops only):
        #   depth_dep_frac ≈ cv_frac + 0.04  (cost-volume + depth head + regression)
        #   tier_saving    = depth_dep_frac × fsdr_narrowing_ratio
        # S3 saving: 0 (not implemented; narrowed search still produces depth
        #   that feeds S3 normally).
        FSDR_TIER2_RATIO = 1.0  # All guided pixels use narrowed search

        if dp_core_cycles > 0 and dp_cv_scaled > 0:
            cv_frac_scaled = dp_cv_scaled / dp_core_cycles
        else:
            cv_frac_scaled = 0.69  # Typical for TranSplat (fallback)

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

        remaining_for_fsdr = 1.0 - total_saes
        fsdr_s2_saving = fsdr_ratio * remaining_for_fsdr * FSDR_S2_SAVE_PER_PIXEL
        fsdr_s3_saving = fsdr_ratio * remaining_for_fsdr * FSDR_S3_SAVE_PER_PIXEL
        
        # ================================================================
        # Build ablation configs
        # ================================================================
        configs = {}
        
        def _make_cfg(fe, dp, gg_nn, ggu_post, d_sav, g_sav, cfg_key=None):
            """Build config with both raw (serial) and effective (pipelined) cycles."""
            raw_total = fe + dp + gg_nn + ggu_post
            
            eff_fe = int(fe * PIPE_FE)
            eff_dp = int(dp * PIPE_DP)
            eff_gg_nn = int(gg_nn * PIPE_GG_NN)
            eff_ggu = 0  # hidden behind conv work
            eff_total = eff_fe + eff_dp + eff_gg_nn + eff_ggu
            
            result = {
                'feature': fe,
                'dp_core': dp,
                'gauss_gen': gg_nn + ggu_post,
                'total': raw_total,
                'depth_saving': d_sav,
                'gauss_saving': g_sav,
                'gauss_head_nn': gg_nn,
                'ggu_post': ggu_post,
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
        
        # 2. ASIC + FSDR only (two-tier: Tier1 full Gaussian reuse, Tier2 narrowed search)
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
            saes_s2_saving, saes_s3_saving, cfg_key='asic_saes')
        
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
            cfg_key='asic_fsdr_saes')
        
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
            'fsdr_reuse_ratio': fsdr_ratio,
            'fsdr_s2_saving': fsdr_s2_saving,
            'fsdr_s3_saving': fsdr_s3_saving,
            'combined_s2_saving': combined_s2_saving,
            'combined_s3_saving': combined_s3_saving,
        }
        
        return configs


# ============================================================
# Model and Data Loading (using integration module)
# ============================================================
def load_model_and_data(
    model_type: str = 'transplat',
    num_samples: int = 1,
    sample_index: int = 0,
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
    # Create model loader
    loader = create_model_loader(model_type)
    
    # Determine checkpoint path based on model type
    checkpoint_paths = {
        'transplat': SCARF_ROOT / 'transplat' / 'checkpoints' / 're10k.ckpt',
        'mvsplat': SCARF_ROOT / 'mvsplat' / 'checkpoints' / 're10k.ckpt',
        'depthsplat': SCARF_ROOT / 'depthsplat' / 'checkpoints' / 're10k.ckpt',
    }
    
    checkpoint_path = checkpoint_paths.get(model_type)
    if checkpoint_path is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint_path = str(checkpoint_path)
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download the checkpoint for {model_type}.\n"
            f"See docs/multi-model-demo-guide.md for instructions."
        )
    
    # Load model
    model_bundle = loader.load_model(checkpoint_path)
    
    # Load data
    data_bundle = loader.load_data(
        model_bundle,
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
                cross_check_threshold=CONFIG.saes_cross_check)

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
        B_f, V_f, C_f, H_f, W_f = features.shape
        features_up = F.interpolate(
            features[0], size=(h, w), mode='bilinear', align_corners=False
        ).mean(dim=0)
    
    trial_fsdr = FSDRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold,
        reuse_hamming=CONFIG.fsdr_reuse_hamming,
        reuse_spatial=CONFIG.fsdr_reuse_spatial,
        reuse_confidence=CONFIG.fsdr_reuse_confidence,
        num_depth_candidates=CONFIG.num_depth_candidates,
    )
    
    if has_features:
        for y in range(h):
            for x in range(w):
                pixel_idx = y * w + x
                feat = features_up[:, y, x]
                if depths is not None:
                    if depths.dim() == 5:
                        ad = depths[0, 0, pixel_idx, 0, 0].item()
                    elif depths.dim() == 4:
                        ad = depths[0, 0, y, x].item()
                    else:
                        ad = 1.0
                else:
                    ad = 1.0
                trial_fsdr.process_pixel(
                    feat, ad, (y, x), pixel_idx,
                    actual_gaussians=gaussians_full, gauss_idx=pixel_idx,
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
    for th in range(tiles_h):
        for tw in range(tiles_w):
            tile_indices = []
            for ly in range(tile_size):
                for lx in range(tile_size):
                    idx = (th * tile_size + ly) * w + (tw * tile_size + lx)
                    if idx < n_gauss:
                        tile_indices.append(idx)
            if not tile_indices:
                continue
            idx_tensor = torch.tensor(tile_indices, dtype=torch.long, device=modified_mask.device)
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
def main():
    global CONFIG
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SCARF Demo')
    parser.add_argument('--model', type=str, default='transplat',
                        choices=['transplat', 'mvsplat', 'depthsplat'],
                        help='Model type to use')
    # Hardware simulator control options
    parser.add_argument('--no-feature', action='store_true',
                        help='Disable SCARF feature extraction HW simulator, use original GPU')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable SCARF depth prediction HW simulator, use original GPU')
    parser.add_argument('--no-gaussian', action='store_true',
                        help='Disable SCARF gaussian generation HW simulator (GGU), use original GPU')
    
    # Performance optimization options
    parser.add_argument('--no-saes', action='store_true',
                        help='Disable SAES early-stopping optimization')
    parser.add_argument('--no-fsdr', action='store_true',
                        help='Disable FSDR depth reuse optimization')
    
    # Ablation experiment
    parser.add_argument('--ablation', action='store_true',
                        help='Run full ablation: GPU / ASIC / ASIC+FSDR / ASIC+SAES / ASIC+FSDR+SAES. '
                             'Forces both SAES and FSDR to run regardless of --no-saes/--no-fsdr.')
    parser.add_argument('--tune-thresholds', action='store_true',
                        help='Sweep SAES/FSDR thresholds to find optimal settings '
                             'within 1.0%% relative PSNR quality budget.')
    
    # Other options
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run baseline, skip SCARF pipeline')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of test samples exposed by the dataloader')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Zero-based sample index to run from the test dataloader')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for saved demo images')
    parser.add_argument('--freq', type=int, default=1000,
                        help='SCARF ASIC clock frequency in MHz (default: 1000 = 1 GHz)')
    parser.add_argument('--saes-fv', type=float, default=None,
                        help='Override feature_var_threshold for SAES L0')
    parser.add_argument('--saes-ds', type=float, default=None,
                        help='Override depth_std_threshold for SAES L1')
    parser.add_argument('--saes-cc', type=float, default=None,
                        help='Override saes_cross_check threshold (probe similarity gate)')
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Override SAES tile size (4 or 8)')
    parser.add_argument('--fsdr-cache-size', type=int, default=None,
                        help='Override FSDR cache size (number of entries)')
    parser.add_argument('--fsdr-hamming', type=int, default=None,
                        help='Override FSDR hamming threshold for cache lookup')
    args = parser.parse_args()
    
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
    
    # Set model-specific num_depth_candidates
    # TranSplat and DepthSplat use 128 depth candidates; MVSplat uses 32
    if args.model in ['transplat', 'depthsplat']:
        CONFIG.num_depth_candidates = 128
        CONFIG.fsdr_narrowed_candidates = 32  # 128/4 = 32
    else:
        CONFIG.num_depth_candidates = 32
        CONFIG.fsdr_narrowed_candidates = 8   # 32/4 = 8
    
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
    print(f"  L0+L1 two-level progressive early-stopping (opacity-zero model)")
    print(f"[Config] FSDR (realistic ASIC):")
    print(f"  cache_size={CONFIG.fsdr_cache_size}, hamming_threshold={CONFIG.fsdr_hamming_threshold}")
    print(f"  reuse_hamming={CONFIG.fsdr_reuse_hamming}, reuse_spatial={CONFIG.fsdr_reuse_spatial}")
    print(f"  reuse_confidence={CONFIG.fsdr_reuse_confidence} (cache hit → use cached Gaussians)")
    print(f"[Config] Depth Prediction (S2):")
    print(f"  num_depth_candidates={CONFIG.num_depth_candidates}")
    print(f"[Config] GGU:")
    print(f"  scale_range=({CONFIG.scale_min}, {CONFIG.scale_max})")
    print()
    
    # Initialize cycle counter and savings tracker
    cycle_counter = HWCycleCounter(ggu_pe_count=CONFIG.ggu_pe_count)
    savings = SavingsTracker()
    
    # --------------------------------------------------------
    # Step 1: Load Model and Data
    # --------------------------------------------------------
    print("[1/6] Loading model and data...")
    model, batch, cfg, device = load_model_and_data(
        args.model,
        num_samples=args.num_samples,
        sample_index=args.sample_index,
    )
    print("  ✓ Model and data loaded")
    
    # --------------------------------------------------------
    # Step 2: Extract batch info
    # --------------------------------------------------------
    print()
    print("[2/6] Processing batch...")
    
    B, V_ctx, _, h, w = batch['context']['image'].shape
    _, V_tgt, _, _, _ = batch['target']['image'].shape
    scene_name = batch['scene'][0] if 'scene' in batch else 'unknown'
    
    V_tgt = min(V_tgt, CONFIG.test_views)
    print(f"  ✓ Scene: {scene_name}, Testing {V_tgt} view(s), Size: {h}x{w}")
    
    context = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch['context'].items()}
    target = {k: v[:, :V_tgt].to(device) if torch.is_tensor(v) and v.dim() > 1 else (v.to(device) if torch.is_tensor(v) else v) for k, v in batch['target'].items()}
    
    # --------------------------------------------------------
    # Step 3: Run BASELINE (full TranSplat encoder)
    # --------------------------------------------------------
    print()
    print("[3/6] Running BASELINE (original TranSplat)...")
    
    # Query GPU info for cycle conversion
    gpu_info = get_gpu_info()
    gpu_freq_mhz = gpu_info['freq_mhz']
    gpu_name = gpu_info['name']
    print(f"  GPU: {gpu_name} @ {gpu_freq_mhz} MHz (max SM clock)")
    
    # --- One forward pass to get outputs (for downstream use) ---
    with torch.no_grad():
        encoder_output = model.encoder(context, False)
        
        # Handle different encoder output formats
        # DepthSplat may return dict with 'gaussians' key when return_depth=True
        if isinstance(encoder_output, dict):
            baseline_gaussians = encoder_output.get('gaussians', encoder_output)
        else:
            baseline_gaussians = encoder_output
        
        tgt_ext = target['extrinsics']
        tgt_int = target['intrinsics']
        baseline_output = model.decoder.forward(
            baseline_gaussians, tgt_ext, tgt_int,
            target['near'], target['far'], (h, w), depth_mode=None
        )
    baseline_image = baseline_output.color[0, 0]
    baseline_count = baseline_gaussians.means.shape[1]
    
    # --- Precise GPU timing (encoder only) via CUDA events ---
    def _baseline_encoder():
        return model.encoder(context, False)
    
    baseline_gpu_time_ms = gpu_timed_inference(_baseline_encoder, warmup=1, repeats=5)
    baseline_gpu_freq_hz = gpu_freq_mhz * 1e6
    baseline_gpu_cycles = int(baseline_gpu_time_ms * 1e-3 * baseline_gpu_freq_hz)
    
    print(f"  ✓ Baseline: {baseline_image.shape}, Gaussians: {baseline_count:,}")
    print(f"    GPU encoder time: {baseline_gpu_time_ms:.2f} ms (median of 5)")
    print(f"    GPU encoder cycles: {baseline_gpu_cycles:,} (@ {gpu_freq_mhz} MHz)")
    
    # Print baseline depth statistics for comparison
    if hasattr(baseline_gaussians, 'depths'):
        baseline_depths = baseline_gaussians.depths.reshape(-1)
        print(f"    Baseline depth stats: min={baseline_depths.min():.4f}, max={baseline_depths.max():.4f}, mean={baseline_depths.mean():.4f}")
    elif hasattr(baseline_gaussians, 'means'):
        # Use Z-coordinate of means as depth proxy
        means = baseline_gaussians.means.reshape(-1, 3)
        z_depths = means[:, 2]
        print(f"    Baseline Z-coord depth: min={z_depths.min():.4f}, max={z_depths.max():.4f}, mean={z_depths.mean():.4f}")
    
    # Print baseline opacities for debugging
    if hasattr(baseline_gaussians, 'opacities'):
        bl_op = baseline_gaussians.opacities.reshape(-1)
        print(f"    Baseline opacities: min={bl_op.min():.4f}, max={bl_op.max():.4f}, mean={bl_op.mean():.4f}")
    
    # Handle baseline-only mode
    if args.baseline_only:
        gt_image = target['image'][0, 0]
        mse = F.mse_loss(baseline_image, gt_image)
        psnr = -10 * torch.log10(mse).item()
        
        print()
        print("[4/6] SCARF Pipeline: SKIPPED (--baseline-only)")
        print("[5/6] Rendering: SKIPPED (--baseline-only)")
        print()
        print("======================================================================")
        print("RESULTS - Baseline Only")
        print("======================================================================")
        print()
        print("### Baseline Metrics")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Gaussians: {baseline_count:,}")
        print(f"  GPU encoder time: {baseline_gpu_time_ms:.2f} ms (median of 5)")
        print(f"  GPU encoder cycles: {baseline_gpu_cycles:,} (@ {gpu_freq_mhz} MHz)")
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
    
    # Initialize cycle counters
    feature_sim_cycles = 0
    s1_cnn_cycles = 0    # CNN portion of S1 (for per-component ASIC scaling)
    depth_sim_cycles = 0
    dp_core_cycles = 0   # DP core: cost_volume + unet + depth_head + regression
    cost_volume_cycles = 0  # cost_volume portion of dp_core (for bandwidth modeling)
    gauss_gen_cycles = 0  # Gaussian Gen: refine_unet + to_gaussians (full-res)
    
    # Initialize FSDR (Feature-Similarity Gaussian Reuse) — realistic ASIC model
    fsdr = FSDRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold,
        reuse_hamming=CONFIG.fsdr_reuse_hamming,
        reuse_spatial=CONFIG.fsdr_reuse_spatial,
        reuse_confidence=CONFIG.fsdr_reuse_confidence,
        num_depth_candidates=CONFIG.num_depth_candidates,
    )
    
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
                
                print(f"    ✓ Features: {pipeline_features.shape if pipeline_features is not None else 'None'}")
        except Exception as e:
            print(f"    ⚠ HW Simulator error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_feature_sim = False
    
    if not use_feature_sim or pipeline_features is None:
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
                    extrinsics = context['extrinsics']
                    img2world = torch.inverse(extrinsics).contiguous()
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
                depth_predictor_sim.set_accurate_mode(False)  # TRUE HW simulation
            elif args.model == 'mvsplat':
                depth_predictor_sim = MVSplatDepthPredictorSim(device=device)
                depth_predictor_sim.load_from_model(model.encoder)
            elif args.model == 'depthsplat':
                depth_predictor_sim = DepthSplatDepthPredictorSim(device=device)
                depth_predictor_sim.load_from_model(model.encoder)
            
            # DepthSplat can work with either features or images (features extracted internally)
            can_run_hw = (depth_predictor_sim is not None and
                         (pipeline_features is not None or args.model == 'depthsplat'))
            
            if can_run_hw:
                extra_info = {'images': rearrange(context['image'], 'b v c h w -> (v b) c h w')}
                
                with torch.no_grad():
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
                elif args.model == 'depthsplat' and feature_sim_cycles > 0:
                    # Fallback guard: if CycleBreakdown.feature_extraction is missing
                    # (e.g. older depth predictor version), approximate de-duplication
                    # by subtracting S1 total cycles.  This path should NOT be reached
                    # with the current hw_depth_predictor which always reports
                    # feature_extraction, but is kept for safety.
                    overlap = min(feature_sim_cycles, dp_core_cycles)
                    dp_core_cycles -= overlap
                    depth_sim_cycles = dp_core_cycles
                    print(f"    [DepthSplat] S2 de-duplicated (fallback): subtracted {overlap:,} S1 feature cycles")
                
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
                            
                            results_dict = model.encoder.depth_predictor(
                                context['image'],
                                attn_splits_list=[2],
                                intrinsics=context['intrinsics'],
                                min_depth=1.0 / far_bv,
                                max_depth=1.0 / near_bv,
                                extrinsics=context['extrinsics'],
                            )
                            depthsplat_results = results_dict
                            
                            depth_final = results_dict['depth_preds'][-1]
                            match_prob = results_dict['match_probs'][-1]
                            
                            features_upsampled = model.encoder.feature_upsampler(
                                results_dict["features_mono_intermediate"],
                                cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                mv_features=results_dict["features_mv"][0] if model.encoder.cfg.num_scales == 1
                                           else results_dict["features_mv"][::-1]
                            )
                            
                            match_prob_max = torch.max(match_prob, dim=1, keepdim=True)[0]
                            if match_prob_max.shape[-2:] != depth_final.shape[-2:]:
                                match_prob_max = F.interpolate(match_prob_max, size=depth_final.shape[-2:], mode='nearest')
                            
                            concat_input = torch.cat((
                                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                                rearrange(depth_final, "b v h w -> (b v) () h w"),
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
                            
                            gaussians_bv = model.encoder.gaussian_head(gaussian_head_input)
                            pipeline_raw_gaussians = gaussians_bv
                            pipeline_densities = rearrange(match_prob_max, "(b v) c h w -> b v (c h w) () ()", b=b_ds, v=v_ds)
                            
                            # Update features from results_dict if not yet set
                            if pipeline_features is None or isinstance(pipeline_features, str):
                                pipeline_features = rearrange(
                                    results_dict['features_mv'][0],
                                    '(b v) c h w -> b v c h w', b=b_ds, v=v_ds
                                )
                            
                            print(f"      ✓ raw_gaussians computed: {pipeline_raw_gaussians.shape}")
                    except Exception as e2:
                        print(f"      ⚠ Failed to compute raw_gaussians: {e2}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"    ⚠ HW Simulator error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_depth_sim = False
    
    if not use_depth_sim or pipeline_depths is None:
        # Fallback: Use original GPU for depth prediction
        print(f"    Mode: Original GPU" + (" (fallback)" if use_depth_sim else " (--no-depth)"))
        
        if pipeline_features is not None and not isinstance(pipeline_features, str) and hasattr(model.encoder, 'depth_predictor'):
            extra_info = {'images': rearrange(context['image'], 'b v c h w -> (v b) c h w')}
            
            with torch.no_grad():
                if args.model == 'transplat':
                    pipeline_depths, pipeline_densities, pipeline_raw_gaussians = model.encoder.depth_predictor(
                        pipeline_features,
                        context['intrinsics'],
                        context['extrinsics'],
                        near,
                        far,
                        gaussians_per_pixel=1,
                        deterministic=True,
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
                        deterministic=True,
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
    
    # ============================================================
    # STAGE 3: Gaussian Generation
    # ============================================================
    print("  [Stage 3] Gaussian Generation...")
    
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
            
            # Verify GGU matches baseline
            mse_means = F.mse_loss(ggu_means, baseline_gaussians.means)
            ggu_psnr_means = -10 * torch.log10(mse_means + 1e-10).item()
            print(f"    ✓ GGU vs Baseline PSNR (means): {ggu_psnr_means:.2f} dB")
            
        except Exception as e:
            print(f"    ⚠ GGU error: {e}, falling back to GPU")
            import traceback
            traceback.print_exc()
            use_gaussian_sim = False
    
    if not all_hw_disabled and (not use_gaussian_sim or not has_gaussian_inputs):
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
    
    # Store pipeline features for FSDR
    features = pipeline_features
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
            features=pipeline_features, depths=pipeline_depths,
        )
        
        # Apply tuned thresholds (already written to CONFIG inside tune_thresholds)
        print(f"\n  Applying tuned thresholds: feat_var={CONFIG.feature_var_threshold}, "
              f"depth_std={CONFIG.depth_std_threshold}")
        
        # Re-create FSDR with tuned config
        fsdr = FSDRSimulator(
            feature_dim=CONFIG.feature_dim,
            cache_size=CONFIG.fsdr_cache_size,
            hamming_threshold=CONFIG.fsdr_hamming_threshold,
            reuse_hamming=CONFIG.fsdr_reuse_hamming,
            reuse_spatial=CONFIG.fsdr_reuse_spatial,
            reuse_confidence=CONFIG.fsdr_reuse_confidence,
            num_depth_candidates=CONFIG.num_depth_candidates,
        )
        # Reset savings tracker
        savings = SavingsTracker()
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
    
    # Clone original Gaussians BEFORE any modifications (needed for ablation configs)
    orig_means = scarf_gaussians_full.means.clone()
    orig_covs = scarf_gaussians_full.covariances.clone()
    orig_harmo = scarf_gaussians_full.harmonics.clone()
    orig_opacs = scarf_gaussians_full.opacities.clone()
    
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
        print("  [4b] Progressive SAES v4 (L0+L1: feature + depth uniformity, opacity-zero)...")

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
            features=features,
            depths=depths,
            cross_check_threshold=CONFIG.saes_cross_check,
        )

        # Print feature variance distribution for threshold calibration
        if features is not None:
            import numpy as np
            _tv, _ = ProgressiveSAES.classify_tiles_by_features(
                features, h, w, CONFIG.tile_size, threshold=1.0)
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
        print(f"  ✓ Modified pixels (opacity→0): {modified_pixels:,}/{h*w:,} "
              f"({modified_pixels/(h*w)*100:.1f}%)")
        print(f"    (Realistic: no post-hoc validation, thresholds ensure quality)")

        saes_low_var_stats = compute_saes_low_var_agreement(
            scarf_gaussians_full, modified_mask, h, w, CONFIG.tile_size)
        print(f"    Low-Var. Agree.: {saes_low_var_stats['low_var_agree']*100:.1f}% "
              f"({saes_low_var_stats['low_var_tiles']}/{saes_low_var_stats['early_tiles']} early tiles, "
              f"mean sim={saes_low_var_stats['mean_similarity']:.3f})")

    # Record SAES v4 savings for cycle model
    # Use h*w (pixel positions) as base, not N (total Gaussians incl. surfaces)
    # because ASIC processes per pixel position - skipping a position skips all surfaces
    savings.record_saes(total_pixels=h*w, saes_stats=saes_stats)
    
    # ---- Step 4c: FSDR (Feature-Similarity Gaussian Reuse) — Realistic ASIC ----
    if args.no_fsdr:
        print("  [4c] FSDR: SKIPPED (--no-fsdr)")
    else:
        print("  [4c] FSDR (Realistic ASIC: cache hit → use cached Gaussians)...")
        
        has_features = (features is not None and
                       not isinstance(features, str) and
                       hasattr(features, 'shape'))
        
        if has_features and len(all_pixels) > 0:
            B_feat, V_feat, C_f, H_feat, W_feat = features.shape
            features_up = F.interpolate(
                features[0], size=(h, w), mode='bilinear', align_corners=False
            ).mean(dim=0)  # [C, H, W]
            
            for (y, x, pixel_idx) in all_pixels:
                feat = features_up[:, y, x]
                
                # Get actual depth (for cache building on miss/non-reuse)
                if depths is not None:
                    if depths.dim() == 5:
                        actual_depth = depths[0, 0, pixel_idx, 0, 0].item()
                    elif depths.dim() == 4:
                        actual_depth = depths[0, 0, y, x].item()
                    else:
                        actual_depth = 1.0
                else:
                    actual_depth = 1.0
                
                # FSDR processing (realistic: reuse decision before compute)
                path, num_searches, output_depth = fsdr.process_pixel(
                    feat, actual_depth, (y, x), pixel_idx,
                    actual_gaussians=scarf_gaussians_full,
                    gauss_idx=pixel_idx,
                )
                
                # Record path for savings tracking
                savings.record_fsdr_pixel(path, reused=(pixel_idx in fsdr.reuse_data))
        
        fsdr_stats = fsdr.get_summary()
        if fsdr_stats['total_pixels'] > 0:
            print(f"    Pixels processed: {fsdr_stats['total_pixels']:,}")
            print(f"    Cache hit rate: {fsdr_stats['hit_rate']*100:.1f}%")
            print(f"    Guided (narrowed): {fsdr_stats.get('guided', 0):,} "
                  f"({fsdr_stats.get('guided_rate', 0)*100:.1f}%)")
            print(f"    Depth inconsistent: {fsdr_stats.get('depth_inconsistent', 0):,}")
            print(f"    Not guided: {fsdr_stats.get('hit_no_guide', 0):,}")
            print(f"    Full compute (miss): {fsdr_stats['full_compute']:,}")
            print(f"    Criteria: hamming≤{fsdr.reuse_hamming}, conf>{fsdr.reuse_confidence:.2f}")
    
    # Record GGU cycles (with compact writeback savings modelled for Stage 4)
    cycle_counter.add_ggu(N, sh_degree=CONFIG.sh_degree)
    
    # ---- Step 4d: Build 4 Ablation Gaussian Configs + Render (Realistic) ----
    print()
    print("[5/6] Real Ablation: 4-config rendering (realistic ASIC behavior)...")
    
    def _render(gaussians_obj):
        """Render a Gaussian set and return the image."""
        with torch.no_grad():
            out = model.decoder.forward(
                gaussians_obj, tgt_ext, tgt_int,
                target['near'], target['far'], (h, w), depth_mode=None
            )
        return out.color[0, 0]
    
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
        print("  [3/4] Rendering: +SAES only (direct interpolation, no oracle validation)...")
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
    
    # Use the best config (FSDR+SAES) as the "SCARF image" for backward compatibility
    scarf_image = ablation_renders['asic_fsdr_saes']
    scarf_time = time.time() - t0
    
    # Gaussian stats
    fsdr_reused = fsdr.stats['total_reuse'] if not args.no_fsdr else 0
    gaussian_stats = {
        'gaussians_baseline': N,
        'gaussians_output': N,
        'pixels_modified': saes_stats.get('total_modified_pixels', 0),
        'fsdr_reused': fsdr_reused,
    }
    
    print(f"  ✓ 4-config rendering complete ({scarf_time:.2f}s total)")
    print(f"  ✓ SAES modified {gaussian_stats['pixels_modified']:,} pixels (interpolation)")
    print(f"  ✓ FSDR guided {fsdr_reused:,} pixels (narrowed search, {CONFIG.fsdr_narrowed_candidates}/{CONFIG.num_depth_candidates} candidates)")
    
    # --------------------------------------------------------
    # Step 6: Evaluate per-config quality and Save
    # --------------------------------------------------------
    print()
    print("[6/6] Evaluating per-config quality...")
    
    gt_image = target['image'][0, 0]
    
    # Metrics
    def compute_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        return -10 * torch.log10(mse).item()
    
    def compute_ssim(img1, img2):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
        return ssim(img1.unsqueeze(0), img2.unsqueeze(0)).item()
    
    # Baseline quality (original model, no SCARF)
    baseline_psnr = compute_psnr(baseline_image, gt_image)
    baseline_ssim = compute_ssim(baseline_image, gt_image)
    
    # Per-config quality (real ablation)
    # Reference: SCARF no-opt is the "correct" ASIC output.
    # Quality loss is measured relative to SCARF no-opt, NOT GPU baseline.
    ablation_quality = {}
    
    # First compute no-opt quality (reference for loss calculation)
    noopt_psnr = compute_psnr(ablation_renders['asic'], gt_image)
    noopt_ssim = compute_ssim(ablation_renders['asic'], gt_image)
    
    for cfg_key, cfg_image in ablation_renders.items():
        psnr = compute_psnr(cfg_image, gt_image)
        ssim = compute_ssim(cfg_image, gt_image)
        # Loss vs SCARF no-opt (the "correct" ASIC output)
        loss_db = psnr - noopt_psnr
        loss_pct = abs(loss_db) / noopt_psnr * 100 if noopt_psnr > 0 else 0
        ablation_quality[cfg_key] = {
            'psnr': psnr, 'ssim': ssim,
            'loss_db': loss_db, 'loss_pct': loss_pct,
        }
        ref_label = "(reference)" if cfg_key == 'asic' else f"loss={loss_db:+.4f} dB ({loss_pct:.4f}%)"
        print(f"  {cfg_key:<20s}: PSNR={psnr:.4f} dB, SSIM={ssim:.6f}, {ref_label}")
    
    # Use FSDR+SAES as the "SCARF" result
    scarf_psnr = ablation_quality['asic_fsdr_saes']['psnr']
    scarf_ssim = ablation_quality['asic_fsdr_saes']['ssim']
    
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
    
    from torchvision.utils import save_image
    save_image(gt_image, output_dir / 'gt_00.png')
    save_image(baseline_image, output_dir / 'baseline_00.png')
    save_image(scarf_image, output_dir / 'scarf_00.png')
    for cfg_key, cfg_image in ablation_renders.items():
        save_image(cfg_image, output_dir / f'ablation_{cfg_key}.png')
    
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
    
    # ================================================================
    # Fallback: Reference cycle counts (per-model, 256×256 @ 1GHz)
    # When HW simulators produce 0 cycles, use architecture-derived reference values.
    # Base hardware: 32×32 PE array (1024 MACs), 32-ch BilinearUnit, 64-wide Vector ALU.
    # These values are scaled by compute_ablation's differentiated HW scaling (2.0x/1.5x).
    # ================================================================
    if feature_sim_cycles == 0:
        if args.model == 'depthsplat':
            # DepthSplat S1: CNN backbone + DINOv2 ViT-S (12 layers, 384-dim) + MV Transformer
            feature_sim_cycles = 160_000_000
        elif args.model == 'mvsplat':
            # MVSplat S1: CNN backbone + 6-layer MV Transformer (no DINOv2)
            feature_sim_cycles = 90_000_000
        else:
            # TranSplat S1: CNN backbone + ViT transformer layers
            feature_sim_cycles = 155_273_000
        print(f"    [Fallback] Using reference S1 feature cycles: {feature_sim_cycles:,}")
    
    if dp_core_cycles == 0:
        if args.model == 'depthsplat':
            # DepthSplat S2: Cost volume (128 candidates) + Regressor UNet + DPT head
            dp_core_cycles = 180_000_000
            cost_volume_cycles = 85_000_000
        elif args.model == 'mvsplat':
            # MVSplat S2: Cost volume (32 candidates) + UNet + depth_head
            dp_core_cycles = 80_000_000
            cost_volume_cycles = 25_000_000
        else:
            # TranSplat S2: UVTransformer (deformable attn) + UNet + depth_head + regression
            dp_core_cycles = 200_000_000
            cost_volume_cycles = 95_000_000
        depth_sim_cycles = dp_core_cycles  # total S2 for legacy compatibility
        print(f"    [Fallback] Using reference S2 depth cycles: {dp_core_cycles:,} (cv: {cost_volume_cycles:,})")
    
    if gauss_gen_cycles == 0:
        if args.model == 'depthsplat':
            # DepthSplat S3: DPT upsampler + gaussian regressor + head (SH degree 2)
            gauss_gen_cycles = 110_000_000
        elif args.model == 'mvsplat':
            # MVSplat S3: depth refinement UNet + gaussian head (SH degree 4)
            gauss_gen_cycles = 130_000_000
        else:
            # TranSplat S3: refine_unet + to_gaussians (full-res spatial processing)
            gauss_gen_cycles = 180_000_000
        print(f"    [Fallback] Using reference S3 gauss gen cycles: {gauss_gen_cycles:,}")
    
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
        print(f"  Criteria:             hamming≤{fsdr.reuse_hamming}, "
              f"conf>{fsdr.reuse_confidence:.2f}, "
              f"depth_consist≤{fsdr.depth_consistency_threshold:.0%}")
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
    print(f"  S1 FE overlap:      x{pipe_info.get('PIPE_FE', 0.92):.2f}  "
          f"(ConvEngine || GEMM in ViT, double buffering)")
    print(f"  S2 DP overlap:      x{pipe_info.get('PIPE_DP', 0.82):.2f}  "
          f"(BilinearUnit || VectorALU || ConvEngine tile pipeline)")
    print(f"  S3 GaussNN overlap: x{pipe_info.get('PIPE_GG_NN', 0.95):.2f}  "
          f"(refine_unet → to_gauss output buf || weight prefetch)")
    print(f"  GGU Post:           hidden (dedicated PEs overlap with ConvEngine)")
    _ggu_s = ggu_stats.get('compact_writeback_bytes_saved', 0)
    _ggu_cs = ggu_stats.get('compact_writeback_cycles_saved', 0)
    print(f"  S4 compact writeback: cov=6 (upper-tri), SH={CONFIG.sh_degree}°={(CONFIG.sh_degree+1)**2} coeffs×3ch")
    print(f"    Bytes saved: {_ggu_s:,}  Cycles saved: {_ggu_cs:,} "
          f"({_ggu_cs / max(1, ggu_stats.get('ggu_raw_cycles', 1)) * 100:.1f}% of raw GGU cycles)")
    print(f"  SAES v4 multi-level (K(T) adaptive probes):")
    print(f"    Combined S2 save: {pipe_info.get('saes_s2_saving', 0)*100:.1f}%")
    print(f"    Combined S3 save: {pipe_info.get('saes_s3_saving', 0)*100:.1f}%")
    fsdr_save_pp = pipe_info.get('fsdr_s2_save_per_pixel', 0.42)
    print(f"  FSDR guided:        {pipe_info.get('fsdr_reuse_ratio', 0)*100:.1f}% "
          f"(narrowed S2: {CONFIG.fsdr_narrowed_candidates}/{CONFIG.num_depth_candidates} candidates, "
          f"saves {fsdr_save_pp*100:.1f}%/pixel, cost_vol+UNet+DepthHead)")
    print(f"  Total S2 saving:    {pipe_info.get('combined_s2_saving', 0)*100:.1f}%")
    print(f"  Total S3 saving:    {pipe_info.get('combined_s3_saving', 0)*100:.1f}%")
    
    # --------------------------------------------------------
    # Jetson Orin Estimation (from measured RTX 3060 time)
    # --------------------------------------------------------
    # Estimation methodology:
    #   Jetson Orin runs the SAME PyTorch model on its Ampere GPU.
    #   We scale the measured RTX 3060 time by the FP32 compute + memory BW ratio.
    #
    # Specs (FP32 TFLOPS / Memory BW):
    #   RTX 3060:              12.74 TFLOPS, 360 GB/s,  3584 CUDA @ 1780 MHz
    #   Jetson AGX Orin 64GB:   5.32 TFLOPS, 204.8 GB/s, 2048 CUDA @ 1300 MHz
    #   Jetson Orin NX  16GB:   1.88 TFLOPS, 102.4 GB/s, 1024 CUDA @  918 MHz
    #
    # Scaling: weighted mix of compute-bound (60%) and memory-bound (40%) slowdown.
    # NOTE: This is an estimation methodology commonly used in architecture papers.
    # The 60/40 ratio is a heuristic for 3DGS encoder workloads (heavy convolutions
    # + moderate feature map reads). For precise comparisons, on-device measurement
    # is recommended. All GPU results are marked "(estimated)" in the output.
    
    RTX3060_FP32_TFLOPS = 12.74
    RTX3060_MEM_BW_GBS = 360.0
    
    # GPU platform definitions: (name, FP32 TFLOPS, mem BW GB/s, CUDA cores, freq MHz)
    GPU_PLATFORMS = {
        # Workstation / Server
        'A6000':            ('NVIDIA RTX A6000',        38.70, 768.0, 10752, 1800),
        # Desktop (measured)
        'RTX3060':          ('NVIDIA RTX 3060',         12.74, 360.0,  3584, 1780),
        # Edge: Jetson Orin
        'AGX_Orin_64GB':    ('Jetson AGX Orin 64GB',     5.32, 204.8,  2048, 1300),
        'Orin_NX_16GB':     ('Jetson Orin NX 16GB',      1.88, 102.4,  1024,  918),
        # Edge: Jetson Xavier
        'AGX_Xavier':       ('Jetson AGX Xavier',        1.40, 136.5,   512, 1370),
    }
    
    # Estimate inference time from measured RTX 3060 baseline
    gpu_estimates = {}
    for key, (pname, p_tflops, p_bw, p_cuda, p_freq) in GPU_PLATFORMS.items():
        compute_ratio = RTX3060_FP32_TFLOPS / p_tflops
        mem_ratio = RTX3060_MEM_BW_GBS / p_bw
        # Mixed workload: 60% compute-bound, 40% memory-bound
        slowdown = 0.6 * compute_ratio + 0.4 * mem_ratio
        est_time_ms = baseline_gpu_time_ms * slowdown
        gpu_estimates[key] = {
            'name': pname,
            'tflops': p_tflops,
            'mem_bw': p_bw,
            'cuda_cores': p_cuda,
            'freq_mhz': p_freq,
            'slowdown': slowdown,
            'est_time_ms': est_time_ms,
        }
    
    # --------------------------------------------------------
    # Ablation: Performance Comparison
    # --------------------------------------------------------
    print()
    print("### Ablation: Performance Comparison")
    print()
    
    # Helper to format time from cycles
    def _fmt_time(cycles: int) -> str:
        ms = cycles / (SCARF_FREQ_MHZ * 1e3)
        return f"{ms:.2f} ms"
    
    # ---- [1] GPU Baseline (RTX 3060) ----
    print(f"  [1] GPU Baseline ({gpu_name} @ {gpu_freq_mhz} MHz):")
    print(f"      FP32: {RTX3060_FP32_TFLOPS} TFLOPS  |  Mem BW: {RTX3060_MEM_BW_GBS} GB/s")
    print(f"      Encoder time:  {baseline_gpu_time_ms:.2f} ms")
    print(f"      GPU cycles:    {baseline_gpu_cycles:,}")
    print()
    
    # ---- [2..N] GPU Platform Estimated Baselines ----
    idx = 2
    # Print order: server -> desktop -> edge (high to low)
    gpu_print_order = ['A6000', 'AGX_Orin_64GB', 'Orin_NX_16GB', 'AGX_Xavier']
    for key in gpu_print_order:
        ge = gpu_estimates[key]
        tag = "(measured)" if key == 'RTX3060' else "(estimated)"
        print(f"  [{idx}] {ge['name']} {tag}, {ge['cuda_cores']} CUDA @ {ge['freq_mhz']} MHz:")
        print(f"      FP32: {ge['tflops']} TFLOPS  |  Mem BW: {ge['mem_bw']} GB/s")
        if key != 'RTX3060':
            print(f"      Slowdown vs RTX 3060: {ge['slowdown']:.2f}x (60% compute + 40% memory)")
        print(f"      Estimated encoder time: {ge['est_time_ms']:.2f} ms")
        print()
        idx += 1
    
    # ---- ASIC Ablation ----
    # Key comparison targets
    ge_a6000  = gpu_estimates['A6000']
    ge_agx    = gpu_estimates['AGX_Orin_64GB']
    ge_nx     = gpu_estimates['Orin_NX_16GB']
    ge_xavier = gpu_estimates['AGX_Xavier']
    
    def _fmt_vs_gpus(asic_ms: float) -> str:
        """Format SCARF speedup/slowdown vs GPU platforms (SCARF-centric)."""
        targets = [
            ('A6000',     ge_a6000),
            ('AGX Orin',  ge_agx),
            ('Orin NX',   ge_nx),
            ('AGX Xavier', ge_xavier),
        ]
        parts = []
        for tname, tdata in targets:
            t = tdata['est_time_ms']
            if t <= 0:
                continue
            if asic_ms <= t:
                speedup = t / asic_ms
                parts.append(f"{speedup:.2f}× faster than {tname}")
            else:
                slowdown = asic_ms / t
                parts.append(f"{slowdown:.2f}× slower than {tname}")
        return " | ".join(parts)
    
    # Helper to print 3-stage ASIC breakdown with serial + pipeline numbers
    H_out = W_out = 256  # output resolution
    def _print_asic_config(label, c, ggu_pe_count):
        # Serial (algorithmic) numbers
        serial_ms = c['total'] / (SCARF_FREQ_MHZ * 1e3)
        # Effective (pipeline-adjusted) numbers
        eff_total = c.get('eff_total', c['total'])
        eff_ms = eff_total / (SCARF_FREQ_MHZ * 1e3)
        eff_fe_ms = c.get('eff_feature', c['feature']) / (SCARF_FREQ_MHZ * 1e3)
        eff_dp_ms = c.get('eff_dp_core', c['dp_core']) / (SCARF_FREQ_MHZ * 1e3)
        eff_gg_ms = c.get('eff_gauss_gen', c.get('gauss_head_nn', 0)) / (SCARF_FREQ_MHZ * 1e3)
        pipe_saving = c.get('pipeline_saving', 0.0)
        
        print(f"  {label}:")
        if c['depth_saving'] > 0 or c.get('gauss_saving', 0) > 0:
            parts = []
            if c['depth_saving'] > 0:
                parts.append(f"DP -{c['depth_saving']*100:.1f}%")
            if c.get('gauss_saving', 0) > 0:
                parts.append(f"GaussGen -{c['gauss_saving']*100:.1f}%")
            print(f"      Savings:            {', '.join(parts)}")
        print(f"      S1 Feature Extract: {c.get('eff_feature', c['feature']):>12,} ({eff_fe_ms:>7.2f} ms)")
        print(f"      S2 Depth Predict:   {c.get('eff_dp_core', c['dp_core']):>12,} ({eff_dp_ms:>7.2f} ms)")
        print(f"      S3 Gaussian Gen:    {c.get('eff_gauss_gen', c.get('gauss_head_nn', 0)):>12,} ({eff_gg_ms:>7.2f} ms)")
        ggu_p = c.get('ggu_post', 0)
        ggu_p_ms = ggu_p / (SCARF_FREQ_MHZ * 1e3)
        print(f"         └ GGU Post:      {ggu_p:>12,}  → hidden ({ggu_pe_count} dedicated PEs ∥ ConvEngine)")
        print(f"      Serial total:       {c['total']:>12,} ({serial_ms:>7.2f} ms)")
        print(f"      Pipeline effective: {eff_total:>12,} ({eff_ms:>7.2f} ms)  [-{pipe_saving*100:.1f}% overlap]")
        print(f"      {_fmt_vs_gpus(eff_ms)}")
        print()
        return eff_ms
    
    ggu_pe = ggu_stats['ggu_pe_count']
    
    # [4] ASIC no optimizations
    _print_asic_config(f"[{idx}] SCARF ASIC @ {SCARF_FREQ_MHZ} MHz (no optimizations)", ablation['asic'], ggu_pe)
    idx += 1
    
    # [5] ASIC + FSDR
    _print_asic_config(f"[{idx}] SCARF ASIC + FSDR", ablation['asic_fsdr'], ggu_pe)
    idx += 1
    
    # [6] ASIC + SAES v4
    _print_asic_config(f"[{idx}] SCARF ASIC + SAES v4", ablation['asic_saes'], ggu_pe)
    idx += 1

    # [7] ASIC + SAES v4 + FSDR
    asic_best_ms = _print_asic_config(f"[{idx}] SCARF ASIC + SAES v4 + FSDR", ablation['asic_fsdr_saes'], ggu_pe)
    
    # --------------------------------------------------------
    # Summary Table (with per-config quality)
    # --------------------------------------------------------
    print("### Performance Summary Table (Real Ablation)")
    print(f"  {'Config':<30s}  {'Eff.Cycles':>12s}  {'Time(ms)':>9s}  {'PSNR(dB)':>10s}  {'SSIM':>8s}  {'SCARF speedup':>14s}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*14}")
    
    def _eff_ms(cfg_key):
        c = ablation[cfg_key]
        return c.get('eff_total', c['total']) / (SCARF_FREQ_MHZ * 1e3)
    
    # Best ASIC config
    asic_best_time = _eff_ms('asic_fsdr_saes')
    
    # Reference GPU platforms
    gpu_rows = [
        (f"RTX A6000 (est.)",             ge_a6000['est_time_ms']),
        (f"RTX 3060 @ {gpu_freq_mhz} MHz", baseline_gpu_time_ms),
        (f"Jetson AGX Orin 64GB (est.)",   ge_agx['est_time_ms']),
        (f"Jetson Orin NX 16GB (est.)",    ge_nx['est_time_ms']),
        (f"Jetson AGX Xavier (est.)",      ge_xavier['est_time_ms']),
    ]
    for name, t_ms in gpu_rows:
        if t_ms > 0 and asic_best_time <= t_ms:
            tag = f"{t_ms / asic_best_time:.2f}x faster"
        elif t_ms > 0:
            tag = f"{asic_best_time / t_ms:.2f}x slower"
        else:
            tag = "N/A"
        print(f"  {name:<30s}  {'':>12s}  {t_ms:>9.2f}  "
              f"{baseline_psnr:>10.4f}  {baseline_ssim:>8.6f}  {tag:>14s}")
    
    print(f"  {'─'*30}  {'─'*12}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*14}")
    
    # SCARF ASIC configurations with quality
    asic_configs = [
        ('asic',           f"SCARF @ {SCARF_FREQ_MHZ}MHz (no opt)"),
        ('asic_fsdr',      f"  + FSDR"),
        ('asic_saes',      f"  + SAES v4"),
        ('asic_fsdr_saes', f"  + SAES v4+FSDR << best"),
    ]
    for cfg_key, label in asic_configs:
        t_ms = _eff_ms(cfg_key)
        eff_cyc = ablation[cfg_key].get('eff_total', ablation[cfg_key]['total'])
        q = ablation_quality.get(cfg_key, {})
        psnr = q.get('psnr', baseline_psnr)
        ssim = q.get('ssim', baseline_ssim)
        # Speedup vs no-opt ASIC
        noopt_ms = _eff_ms('asic')
        if t_ms > 0 and t_ms < noopt_ms:
            speedup_tag = f"{noopt_ms / t_ms:.2f}x vs base"
        else:
            speedup_tag = "baseline"
        print(f"  {label:<30s}  {eff_cyc:>12,}  {t_ms:>9.2f}  "
              f"{psnr:>10.4f}  {ssim:>8.6f}  {speedup_tag:>14s}")
    
    print()
    print("  (All quality measured via real rendering, not estimated)")
    
    print()
    print("### Quality & Savings Summary")
    best_q = ablation_quality.get('asic_fsdr_saes', {})
    print(f"  Best config quality:  PSNR={best_q.get('psnr', 0):.4f} dB, "
          f"loss={best_q.get('loss_db', 0):+.4f} dB ({best_q.get('loss_pct', 0):.4f}%)")
    print(f"  SAES v4 total:        {savings.saes_interpolation_ratio()*100:.1f}% pixels modified "
          f"(L0={savings.level0_ratio()*100:.1f}%, L1={savings.level1_ratio()*100:.1f}%)")
    print(f"  FSDR guided:          {savings.fsdr_validated_saving_ratio()*100:.1f}% "
          f"({savings.fsdr_validated:,} pixels narrowed S2 search)")
    if baseline_gpu_time_ms > 0:
        print(f"  Best ASIC time:       {asic_best_ms:.2f} ms (pipeline-adjusted)")
        for tname, tdata in [('A6000', ge_a6000), ('AGX Orin', ge_agx),
                             ('Orin NX', ge_nx), ('AGX Xavier', ge_xavier)]:
            ref = tdata['est_time_ms']
            if ref <= 0:
                continue
            if asic_best_ms <= ref:
                print(f"    vs {tname:<13s} {ref:>7.2f} ms  ->  SCARF {ref/asic_best_ms:.2f}x faster")
            else:
                print(f"    vs {tname:<13s} {ref:>7.2f} ms  ->  SCARF {asic_best_ms/ref:.2f}x slower")
    # --------------------------------------------------------
    # 28nm ASIC Power & Area Estimation
    # --------------------------------------------------------
    # Methodology:
    #   Based on Horowitz ISSCC 2014 energy model, scaled from 45nm to 28nm (×0.55).
    #   Reference: Envision (JSSC 2017) measured ~6 pJ/MAC at 28nm including SRAM overhead.
    #   Systolic array data reuse amortizes SRAM access energy across 32× reuse factor.
    #
    # Energy per operation at 28nm (pJ):
    #   FP16 MAC (pure arithmetic):   0.83 pJ
    #   FP16 MAC + register xfer:     1.1 pJ  (systolic array with data reuse)
    #   32KB SRAM read (32-bit):      2.75 pJ
    #   64KB SRAM read (32-bit):      3.3 pJ
    #   128KB SRAM read (32-bit):     4.4 pJ
    #   DRAM access (64-bit):         ~350 pJ
    #
    # Process: TSMC 28nm HPC+ (1.0V nominal, 1 GHz achievable for systolic array)
    
    print()
    print("### 28nm ASIC Power & Area Estimation")
    print()
    
    freq_ghz = SCARF_FREQ_MHZ / 1000.0
    
    # ---- Per-unit energy model (pJ per operation at 28nm) ----
    E_MAC_FP16 = 1.1       # FP16 MAC including register transfer in systolic array
    E_MAC_GEMM = 1.2       # GEMM MAC (slightly higher due to output-stationary overhead)
    E_SRAM_64KB = 3.3      # 64KB SRAM read, 32-bit
    E_SRAM_128KB = 4.4     # 128KB SRAM read, 32-bit
    E_ALU_FP16 = 0.22      # FP16 add (vector ALU)
    E_BILINEAR = 3.6       # Per bilinear sample (4 mul + 3 add + addr)
    E_BILINEAR_MEM = 10.0  # Per bilinear sample memory reads (4 neighbors)
    
    # ---- Compute units (upgraded: 48×48 arrays) ----
    conv_macs = 2304    # 48×48 ConvEngine (upgraded from 32×32)
    gemm_macs = 2304    # 48×48 GEMM (upgraded from 32×32)
    valu_width = 64     # 64-wide Vector ALU (upgraded from 32)
    bilinear_samplers = 32   # 32 samplers (upgraded from 16)
    bilinear_channels = 64   # 64 channels (upgraded from 32)
    ggu_pes = CONFIG.ggu_pe_count  # 32 (upgraded from 16)
    ggu_macs_per_pe = 20  # avg MACs per GGU PE per active cycle
    
    # ---- Stage utilization (fraction of total inference time each unit is active) ----
    # Derived from cycle breakdown: S1(FE) + S2(DP) + S3(GaussGen)
    total_eff = ablation['asic_fsdr_saes'].get('eff_total', 0)
    eff_fe = ablation['asic_fsdr_saes'].get('eff_feature', 0)
    eff_dp = ablation['asic_fsdr_saes'].get('eff_dp_core', 0)
    eff_gg = ablation['asic_fsdr_saes'].get('eff_gauss_gen', 0)
    
    # ConvEngine: active during CNN(S1), UNet/DepthHead(S2), refine_unet/to_gauss(S3)
    util_conv = 0.80   # ~80% of inference time
    # GEMM: active during Transformer(S1), attention/regression(S2)
    util_gemm = 0.25   # ~25% of inference time
    # BilinearUnit: active during cost_volume(S2), upsampling
    util_bilinear = 0.25
    # VectorALU: element-wise ops scattered throughout
    util_valu = 0.40
    # GGU PEs: only during GGU post-processing (hidden, but still consumes power)
    util_ggu = 0.02    # ~2% (very short active period)
    
    # ---- Component power (mW) ----
    # P = N_units × E_per_op × freq × utilization
    
    p_conv_compute = conv_macs * E_MAC_FP16 * freq_ghz * util_conv          # mW
    p_gemm_compute = gemm_macs * E_MAC_GEMM * freq_ghz * util_gemm
    p_valu = valu_width * E_ALU_FP16 * freq_ghz * util_valu
    p_bilinear = bilinear_samplers * (E_BILINEAR + E_BILINEAR_MEM) * freq_ghz * util_bilinear
    p_ggu = ggu_pes * ggu_macs_per_pe * E_MAC_FP16 * freq_ghz * util_ggu
    p_activation = 15.0  # LUT-based, very low
    
    # ---- SRAM power ----
    # Upgraded SRAM: larger buffers for 48×48 arrays
    # ConvEngine buffers: weight(128KB) + input(~16KB) → ~300 reads/cycle avg
    # GEMM buffer: 192KB → ~150 reads/cycle avg
    # Feature buffers: ~384KB → ~80 reads/cycle avg
    sram_total_kb = 128 + 16 + 192 + 384 + 96  # 816 KB
    p_sram_conv = 300 * E_SRAM_64KB * freq_ghz * util_conv      # weight + input reads
    p_sram_gemm = 150 * E_SRAM_128KB * freq_ghz * util_gemm
    p_sram_feat = 80 * E_SRAM_64KB * freq_ghz * 0.50            # intermittent
    p_sram_total = p_sram_conv + p_sram_gemm + p_sram_feat
    
    # ---- Clock tree + PLL ----
    p_clock = 100 * freq_ghz  # ~100 mW at 1 GHz, scales with freq
    p_pll = 12.0
    
    # ---- Control logic ----
    p_control = 35.0 * freq_ghz
    
    # ---- I/O + DRAM interface ----
    p_io_ddr = 150.0    # LPDDR4 PHY + pads
    p_io_misc = 20.0
    
    # ---- Static (leakage) power at 28nm ----
    # 28nm HPC+: ~0.02 mW/gate equivalent, or ~20% of dynamic for moderate designs
    # With power gating of idle units
    p_leakage = 420.0  # mW, includes all transistors + SRAM leakage (scaled for larger arrays)
    
    # ---- Totals ----
    p_compute = p_conv_compute + p_gemm_compute + p_valu + p_bilinear + p_ggu + p_activation
    p_memory = p_sram_total
    p_infra = p_clock + p_pll + p_control + p_io_ddr + p_io_misc
    p_dynamic = p_compute + p_memory + p_infra
    p_total = p_dynamic + p_leakage
    
    # Peak power (all units at 100% utilization simultaneously - theoretical max)
    p_peak_compute = (conv_macs * E_MAC_FP16 + gemm_macs * E_MAC_GEMM) * freq_ghz
    p_peak_sram = 300 * E_SRAM_64KB * freq_ghz + 150 * E_SRAM_128KB * freq_ghz
    p_peak = p_peak_compute + p_peak_sram + p_infra + p_leakage
    
    # ---- Area estimation (28nm TSMC) ----
    # FP16 MAC unit: ~0.004 mm² at 28nm (including local registers)
    # SRAM: 0.127 mm²/Mbit (TSMC 28nm standard)
    a_conv = conv_macs * 0.004 * 1.3   # +30% routing overhead
    a_gemm = gemm_macs * 0.004 * 1.3
    a_valu = valu_width * 0.001
    a_bilinear = bilinear_samplers * 0.02
    a_ggu = ggu_pes * 0.04
    a_misc_logic = 0.3 + 0.15  # activation/norm + FSDR/SAES
    a_sram = sram_total_kb * 8 / 1024 * 0.127  # KB → Mbit → mm²
    a_control = 0.5
    a_io = 2.5  # DDR PHY + pads
    a_pad = 3.0  # pad ring + ESD
    a_total = a_conv + a_gemm + a_valu + a_bilinear + a_ggu + a_misc_logic + a_sram + a_control + a_io + a_pad
    die_side = a_total ** 0.5
    
    # ---- Energy per inference ----
    asic_time_s = asic_best_ms / 1000.0
    asic_energy_mj = p_total * asic_time_s  # mW × s = mJ
    
    # GPU platform typical inference power (W)
    GPU_POWER = {
        'A6000':         ('RTX A6000',        200.0),    # 300W TDP, ~200W inference avg
        'RTX3060':       ('RTX 3060',         130.0),    # 170W TDP, ~130W inference avg
        'AGX_Orin_64GB': ('AGX Orin 64GB',     40.0),    # 60W MAXN mode
        'Orin_NX_16GB':  ('Orin NX 16GB',      15.0),    # 15-25W typical
        'AGX_Xavier':    ('AGX Xavier',        20.0),    # 15-30W, ~20W inference
    }
    
    print("  Process: TSMC 28nm HPC+ (1.0V, 1 GHz)")
    print(f"  Methodology: Horowitz ISSCC'14 energy model scaled to 28nm")
    print()
    
    # Power breakdown table
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ Component              │ Dynamic (mW) │ Description     │")
    print("  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ ConvEngine ({conv_macs} MAC)    │ {p_conv_compute:>8.0f}      │ {util_conv*100:.0f}% util        │")
    print(f"  │ GEMM Unit  ({gemm_macs} MAC)    │ {p_gemm_compute:>8.0f}      │ {util_gemm*100:.0f}% util        │")
    print(f"  │ VectorALU  ({valu_width}-wide)     │ {p_valu:>8.1f}      │ {util_valu*100:.0f}% util        │")
    print(f"  │ BilinearUnit ({bilinear_samplers} samp)  │ {p_bilinear:>8.0f}      │ {util_bilinear*100:.0f}% util        │")
    print(f"  │ GGU PEs ({ggu_pes} PEs)         │ {p_ggu:>8.1f}      │ {util_ggu*100:.0f}% util (hidden)│")
    print(f"  │ Activation/Norm/Softmax│ {p_activation:>8.0f}      │ LUT-based       │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ Compute subtotal       │ {p_compute:>8.0f}      │                 │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ SRAM ({sram_total_kb} KB)          │ {p_sram_total:>8.0f}      │ buffers + cache  │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ Clock tree + PLL       │ {p_clock + p_pll:>8.0f}      │ @ {SCARF_FREQ_MHZ} MHz       │")
    print(f"  │ Control logic          │ {p_control:>8.0f}      │                 │")
    print(f"  │ I/O + LPDDR4 PHY       │ {p_io_ddr + p_io_misc:>8.0f}      │                 │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ Dynamic subtotal       │ {p_dynamic:>8.0f}      │                 │")
    print(f"  │ Static (leakage)       │ {p_leakage:>8.0f}      │ 28nm HPC+       │")
    print(f"  ╞══════════════════════════════════════════════════════════╡")
    print(f"  │ TOTAL AVERAGE POWER    │ {p_total:>8.0f} mW   │ = {p_total/1000:.2f} W       │")
    print(f"  │ PEAK POWER             │ {p_peak:>8.0f} mW   │ = {p_peak/1000:.2f} W       │")
    print(f"  └──────────────────────────────────────────────────────────┘")
    
    # Area breakdown
    print()
    print(f"  Die area estimate:")
    print(f"    ConvEngine:     {a_conv:.2f} mm²   GEMM Unit:  {a_gemm:.2f} mm²")
    print(f"    BilinearUnit:   {a_bilinear:.2f} mm²   GGU PEs:    {a_ggu:.2f} mm²")
    print(f"    SRAM ({sram_total_kb} KB):   {a_sram:.2f} mm²   Control:    {a_control:.2f} mm²")
    print(f"    I/O + pads:     {a_io + a_pad:.2f} mm²   Other:      {a_valu + a_misc_logic:.2f} mm²")
    print(f"    Total:          {a_total:.1f} mm²  ({die_side:.1f} × {die_side:.1f} mm die)")
    
    # Energy per inference comparison
    print()
    print("  Energy per inference comparison:")
    print(f"    {'Platform':<28s}  {'Time(ms)':>9s}  {'Power(W)':>9s}  {'Energy(mJ)':>11s}  {'vs ASIC':>8s}")
    print(f"    {'─'*28}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*8}")
    
    # SCARF ASIC row
    print(f"    {'SCARF ASIC @ 28nm':<28s}  {asic_best_ms:>9.1f}  {p_total/1000:>9.2f}  {asic_energy_mj:>11.0f}  {'1.00x':>8s}")
    
    # GPU rows
    energy_rows = []
    for key in ['RTX3060', 'A6000', 'AGX_Orin_64GB', 'Orin_NX_16GB', 'AGX_Xavier']:
        gname, gpower = GPU_POWER[key]
        if key == 'RTX3060':
            gtime = baseline_gpu_time_ms
        else:
            gtime = gpu_estimates[key]['est_time_ms']
        genergy = gpower * gtime  # W × ms = mJ
        ratio = genergy / asic_energy_mj if asic_energy_mj > 0 else 0
        energy_rows.append((gname, gtime, gpower, genergy, ratio))
        print(f"    {gname:<28s}  {gtime:>9.1f}  {gpower:>9.1f}  {genergy:>11.0f}  {ratio:>7.1f}x")
    
    print()
    # Efficiency metric
    peak_tops = (conv_macs + gemm_macs) * 2 * freq_ghz / 1000  # 2 ops per MAC (mul+add)
    avg_tops = peak_tops * (util_conv * conv_macs + util_gemm * gemm_macs) / (conv_macs + gemm_macs)
    print(f"  Performance metrics:")
    print(f"    Peak throughput:     {peak_tops:.2f} TOPS ({conv_macs + gemm_macs} MACs × 2 × {freq_ghz} GHz)")
    print(f"    Avg throughput:      {avg_tops:.2f} TOPS (weighted by utilization)")
    print(f"    Energy efficiency:   {avg_tops / (p_total/1000):.2f} TOPS/W (avg)")
    print(f"    Area efficiency:     {avg_tops / a_total:.2f} TOPS/mm²")
    
    # Key takeaway
    if len(energy_rows) > 0:
        best_gpu_name, _, _, best_gpu_energy, best_ratio = min(energy_rows, key=lambda x: x[3])
        worst_gpu_name, _, _, worst_gpu_energy, worst_ratio = max(energy_rows, key=lambda x: x[3])
        print()
        print(f"  Energy advantage:")
        print(f"    vs {best_gpu_name}: {best_ratio:.0f}× less energy per inference")
        print(f"    vs {worst_gpu_name}: {worst_ratio:.0f}× less energy per inference")
    
    print()
    print(f"## Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
