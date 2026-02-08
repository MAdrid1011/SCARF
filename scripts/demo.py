#!/usr/bin/env python3
"""
SCARF Demo - Hardware Simulator for 3D Gaussian Splatting Encoders

SCARF (Scene-Adaptive Cost-volume Accelerator with Reuse Framework) is a
hardware-realizable accelerator for 3D Gaussian Splatting encoders.

This demo shows SCARF's two key optimizations:
1. SAES (Scene-Adaptive Early-Stopping): Skip depth search for homogeneous tiles
2. FSGR (Feature-Similarity Depth Reuse): Cache and reuse depth for similar pixels

Data Flow:
    [Neural Network Backbone] → features
           ↓
    [SCARF SAES] → tile decisions (early-stop vs continue)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Early-Stop     Continue
    ↓             ↓
Skip S2     [S2 Depth + FSGR] → depths
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
- FSGR: Cache hit rate, Memory access reduction

Usage:
    python demo.py [--model MODEL_TYPE] [OPTIONS]

Supported Models:
    - transplat (default)
    - mvsplat
    - depthsplat

Fallback Options (for debugging):
    --no-gaussian     Disable SCARF gaussian generation HW simulator (GGU), use original GPU
    --no-saes         Disable SAES early-stopping
    --no-fsgr         Disable FSGR depth reuse
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
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import argparse

# SCARF imports
from adapters import create_adapter, BaseAdapter
from integration import create_model_loader, ModelBundle, DataBundle
from ggu import GGUProcessor, GGUConfig
from fsgr import FSGRSimulator
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
    
    # ---- SAES v3 multi-level config ----
    # L0 now uses 4-probe interpolation (not 1-probe replication) → better quality
    # This allows more aggressive thresholds while maintaining quality
    tile_size: int = 4
    saes_threshold: float = 0.995     # Level 2: Gaussian similarity threshold
    saes_cov_safety: float = 1.02     # Safety factor for interpolated covariances
    feature_var_threshold: float = 0.012  # Level 0: feature variance (4-probe, can be aggressive)
    depth_std_threshold: float = 0.005    # Level 1: relative depth std threshold
    saes_cross_check: float = 0.015       # Probe cross-check error threshold
    
    # ---- FSGR config — Depth-Only Reuse (Realistic ASIC) ----
    # In real ASIC: cache hit → reuse cached DEPTH (skip S2 only)
    # S3 still runs with cached depth → only means/positions affected
    # Quality impact is proportional to depth error (very small for matched pixels)
    # This allows much more aggressive reuse criteria than full-Gaussian reuse
    fsgr_cache_size: int = 512
    fsgr_hamming_threshold: int = 4       # Cache lookup hamming range
    fsgr_reuse_hamming: int = 3           # Moderate hamming (depth-only is safe)
    fsgr_reuse_spatial: int = 12          # Moderate spatial distance
    fsgr_reuse_confidence: float = 0.80   # Min peak_prob for reuse decision
    
    # Depth prediction config (will be overridden based on model type)
    num_depth_candidates: int = 32  # Default for MVSplat; Transplat/DepthSplat use 128
    feature_dim: int = 128
    
    # GGU config (defaults, overridden by adapter)
    scale_min: float = 0.5
    scale_max: float = 15.0
    sh_degree: int = 4
    
    # Test config
    test_views: int = 1
    
    def apply_adapter_overrides(self, adapter: BaseAdapter):
        """Apply model-specific configuration from adapter."""
        fsgr_overrides = adapter.get_fsgr_config_overrides()
        saes_overrides = adapter.get_saes_config_overrides()
        scale_min, scale_max = adapter.get_scale_range()
        
        # Apply overrides
        if 'hamming_threshold' in fsgr_overrides:
            self.fsgr_hamming_threshold = fsgr_overrides['hamming_threshold']
        if 'reuse_hamming' in fsgr_overrides:
            self.fsgr_reuse_hamming = fsgr_overrides['reuse_hamming']
        if 'reuse_spatial' in fsgr_overrides:
            self.fsgr_reuse_spatial = fsgr_overrides['reuse_spatial']
        if 'early_stop_threshold' in saes_overrides:
            self.saes_threshold = saes_overrides['early_stop_threshold']
        
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
    
    def add_ggu(self, num_gaussians: int = 1):
        """Record cycles for GGU gaussian generation."""
        cycles_per_gaussian = sum(self.GGU_CYCLES.values())
        total_cycles = cycles_per_gaussian * num_gaussians
        self._ggu_raw_cycles += total_cycles
        self._ggu_elements += num_gaussians
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
        }


# ============================================================
# Savings Tracker (SAES + FSGR cycle savings model)
# ============================================================
class SavingsTracker:
    """
    Track cycle savings from SAES v3 (multi-level) and FSGR optimizations.
    
    v3 changes:
    - SAES v3: 3-level tile optimization (all levels use 4-probe interpolation)
        Level 0: Feature-uniform tiles (4 probes, 12 interpolated) → saves 75% S2+S3
        Level 1: Depth-uniform tiles (4 probes, 12 interpolated) → saves 75% S2+S3
        Level 2: Gaussian-similar tiles (4 probes, 12 interpolated) → saves 75% S3
    - FSGR: Feature-Similarity Gaussian Reuse. Validated cache hits skip
        FULL S2+S3 per pixel (not just cost_volume like FSGR v2).
    """
    
    LIGHT_VERIFY_COST_RATIO = 0.04
    
    def __init__(self):
        self.total_pixels = 0
        # SAES v3 multi-level
        self.level0_pixels = 0   # Interpolated from 4 probes (12 per tile, v2)
        self.level1_pixels = 0   # Interpolated from 4 probes, depth-based (12 per tile)
        self.level2_pixels = 0   # Interpolated from 4 probes, Gaussian-based (12 per tile)
        self.saes_interpolated_pixels = 0  # Total modified (backward compat)
        # FSGR
        self.fsgr_direct_reuse = 0
        self.fsgr_interpolation = 0
        self.fsgr_light_verify = 0
        self.fsgr_full_search = 0
        self.fsgr_validated = 0
        self.fsgr_rejected = 0
    
    def record_saes(self, total_pixels: int, saes_stats: Dict):
        """Record SAES v3 multi-level statistics (uses validated counts)."""
        self.total_pixels = total_pixels
        # Use validated modified pixels if available (post-validation)
        validated = saes_stats.get('validated_modified_pixels', None)
        if validated is not None:
            # Distribute validated pixels proportionally across levels
            raw_total = (saes_stats.get('level0_pixels', 0) +
                         saes_stats.get('level1_pixels', 0) +
                         saes_stats.get('level2_pixels', 0))
            if raw_total > 0:
                ratio = validated / raw_total
            else:
                ratio = 0.0
            self.level0_pixels = int(saes_stats.get('level0_pixels', 0) * ratio)
            self.level1_pixels = int(saes_stats.get('level1_pixels', 0) * ratio)
            self.level2_pixels = int(saes_stats.get('level2_pixels', 0) * ratio)
        else:
            self.level0_pixels = saes_stats.get('level0_pixels', 0)
            self.level1_pixels = saes_stats.get('level1_pixels', 0)
            self.level2_pixels = saes_stats.get('level2_pixels', 0)
        self.saes_interpolated_pixels = (self.level0_pixels +
                                          self.level1_pixels +
                                          self.level2_pixels)
    
    def record_fsgr_pixel(self, path: str, reused: bool = False, validated: bool = False):
        """Record one pixel's FSGR path with reuse status.
        
        Args:
            path: Decision path ('reuse', 'hit_no_reuse', 'full_compute', etc.)
            reused: True if this pixel used cached Gaussians (skipped S2+S3)
            validated: Deprecated alias for reused (backward compat)
        """
        if path in ('reuse', 'guided'):
            self.fsgr_direct_reuse += 1
        elif path in ('hit_no_reuse', 'hit_no_guide', 'full_compute', 'full_search'):
            self.fsgr_full_search += 1
        else:
            self.fsgr_full_search += 1
        if reused or validated:
            self.fsgr_validated += 1
    
    @property
    def fsgr_total(self) -> int:
        """Total pixels processed by FSGR."""
        return (self.fsgr_direct_reuse + self.fsgr_interpolation +
                self.fsgr_light_verify + self.fsgr_full_search)
    
    # --- Per-level saving ratios ---
    def level0_ratio(self) -> float:
        """Fraction of pixels replicated by Level 0 (feature-uniform)."""
        return self.level0_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def level1_ratio(self) -> float:
        """Fraction of pixels interpolated by Level 1 (depth-uniform)."""
        return self.level1_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def level2_ratio(self) -> float:
        """Fraction of pixels interpolated by Level 2 (Gaussian-similar)."""
        return self.level2_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def saes_interpolation_ratio(self) -> float:
        """Total fraction of pixels modified by SAES (all levels)."""
        return self.saes_interpolated_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def fsgr_validated_saving_ratio(self) -> float:
        """Fraction of ALL pixels validated as fully skippable by FSGR (S2+S3)."""
        return self.fsgr_validated / self.total_pixels if self.total_pixels > 0 else 0.0
    
    
    def compute_ablation(self, feature_cycles: int, depth_cycles: int,
                         ggu_cycles: int,
                         dp_core_cycles: int = 0,
                         gauss_gen_cycles: int = 0,
                         cost_volume_cycles: int = 0,
                         ablation_quality: Dict = None) -> Dict[str, Dict]:
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
          SAES Level 0: Feature-uniform tiles → 4 probes, 12 interpolated (v2: bilinear)
                        Saves 75% of S2+S3 per tile. Decision: S1 feature variance + cross-check.
          SAES Level 1: Depth-uniform tiles → 4 probes S2, interpolate 12 in S3
                        Saves 75% of S2+S3 per tile. Decision: probe depth uniformity + cross-check.
          SAES Level 2: Gaussian-similar tiles → 4 probes, interpolate 12 in S3
                        Saves 75% of S3 per tile. Decision: probe Gaussian similarity + cross-check.
          FSGR (Narrowed Search): Guided pixels search D/4 depth candidates.
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
        
        # Multi-level SAES ratios (fraction of total pixels)
        l0_ratio = self.level0_ratio()   # Feature-uniform: 75% saving on S2+S3 (4-probe)
        l1_ratio = self.level1_ratio()   # Depth-uniform: 75% saving on S2+S3
        l2_ratio = self.level2_ratio()   # Gaussian-similar: 75% saving on S3 only
        total_saes = l0_ratio + l1_ratio + l2_ratio
        
        # FSGR: fraction of remaining (non-SAES) pixels that get guided search
        fsgr_ratio = self.fsgr_validated_saving_ratio()
        
        if dp_core_cycles == 0 and gauss_gen_cycles == 0:
            dp_core_cycles = depth_cycles
            gauss_gen_cycles = 0
        
        # ================================================================
        # Apply differentiated hardware scaling per stage
        # ================================================================
        # S1 (Feature Extraction): CNN convolutions + ViT GEMM → compute-bound
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
        # The pixel ratios already represent the fraction of SKIPPED pixels:
        #   l0_ratio = (num_L0_tiles * 12) / total_pixels  [12 interpolated per tile]
        #   l1_ratio = (num_L1_tiles * 12) / total_pixels  [12 interpolated per tile]
        #   l2_ratio = (num_L2_tiles * 12) / total_pixels  [12 interpolated per tile]
        #
        # L0 & L1 tiles: skipped pixels bypass BOTH S2 and S3
        # L2 tiles: S2 ran for all 16 pixels, only S3 skipped
        saes_s2_saving = l0_ratio + l1_ratio       # L0+L1 skip S2
        saes_s3_saving = l0_ratio + l1_ratio + l2_ratio  # All levels skip S3
        
        # ================================================================
        # FSGR savings: cost_volume-only model (conservative)
        # ================================================================
        # Narrowed search reduces depth candidates from D to D/4 (e.g., 128→32).
        # This directly reduces cost_volume computation (per-pixel per-candidate).
        #
        # Only cost_volume is reduced. U-Net, depth_head, and regression process
        # the full spatial resolution regardless of per-pixel candidate count.
        # This is the most defensible model: savings = cv_fraction * 0.75.
        #
        # No separate "bandwidth bonus" — the cost_volume cycle reduction already
        # includes fewer memory reads (each candidate requires feature warping).
        if dp_core_cycles > 0 and dp_cv_scaled > 0:
            cv_frac_scaled = dp_cv_scaled / dp_core_cycles
        else:
            cv_frac_scaled = 0.69  # Typical for Transplat (fallback)
        
        FSGR_S2_SAVE_PER_PIXEL = cv_frac_scaled * 0.75  # cost_volume-only
        remaining_for_fsgr = 1.0 - total_saes
        fsgr_s2_saving = fsgr_ratio * remaining_for_fsgr * FSGR_S2_SAVE_PER_PIXEL
        fsgr_s3_saving = 0.0  # S3 unchanged
        
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
        
        # 2. ASIC + FSGR only (cost_volume-only savings per guided pixel)
        fsgr_alone_s2 = fsgr_ratio * FSGR_S2_SAVE_PER_PIXEL
        dp_fsgr = int(dp_core_cycles * (1.0 - fsgr_alone_s2))
        gh_fsgr = gauss_gen_cycles   # S3 unchanged
        ggu_fsgr = ggu_cycles        # GGU unchanged
        configs['asic_fsgr'] = _make_cfg(
            feature_cycles, dp_fsgr, gh_fsgr, ggu_fsgr,
            fsgr_alone_s2, 0.0, cfg_key='asic_fsgr')
        
        # 3. ASIC + SAES only (multi-level)
        dp_saes = int(dp_core_cycles * (1.0 - saes_s2_saving))
        gh_saes = int(gauss_gen_cycles * (1.0 - saes_s3_saving))
        ggu_saes = int(ggu_cycles * (1.0 - saes_s3_saving))
        configs['asic_saes'] = _make_cfg(
            feature_cycles, dp_saes, gh_saes, ggu_saes,
            saes_s2_saving, saes_s3_saving, cfg_key='asic_saes')
        
        # 4. ASIC + SAES + FSGR (full optimization)
        combined_s2_saving = saes_s2_saving + fsgr_s2_saving
        combined_s3_saving = saes_s3_saving + fsgr_s3_saving  # fsgr_s3_saving=0
        dp_both = int(dp_core_cycles * (1.0 - combined_s2_saving))
        dp_both = max(0, dp_both)
        gh_both = int(gauss_gen_cycles * (1.0 - combined_s3_saving))
        gh_both = max(0, gh_both)
        ggu_both = int(ggu_cycles * (1.0 - combined_s3_saving))
        ggu_both = max(0, ggu_both)
        configs['asic_fsgr_saes'] = _make_cfg(
            feature_cycles, dp_both, gh_both, ggu_both,
            combined_s2_saving, combined_s3_saving,
            cfg_key='asic_fsgr_saes')
        
        # Store pipeline factors and multi-level info
        configs['_pipeline'] = {
            'PIPE_FE': PIPE_FE,
            'PIPE_DP': PIPE_DP,
            'PIPE_GG_NN': PIPE_GG_NN,
            'GGU_hidden': True,
            'HW_SCALE_C': HW_SCALE_C,
            'HW_SCALE_M': HW_SCALE_M,
            'cv_frac_scaled': cv_frac_scaled,
            'fsgr_s2_save_per_pixel': FSGR_S2_SAVE_PER_PIXEL,
            'saes_l0_ratio': l0_ratio,
            'saes_l1_ratio': l1_ratio,
            'saes_l2_ratio': l2_ratio,
            'saes_total_ratio': total_saes,
            'saes_s2_saving': saes_s2_saving,
            'saes_s3_saving': saes_s3_saving,
            'fsgr_reuse_ratio': fsgr_ratio,
            'fsgr_s2_saving': fsgr_s2_saving,
            'fsgr_s3_saving': fsgr_s3_saving,
            'combined_s2_saving': combined_s2_saving,
            'combined_s3_saving': combined_s3_saving,
        }
        
        return configs


# ============================================================
# Model and Data Loading (using integration module)
# ============================================================
def load_model_and_data(model_type: str = 'transplat'):
    """
    Load model and data using the model loader abstraction.
    
    Args:
        model_type: Type of model ('transplat', 'mvsplat', 'depthsplat')
    
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
    data_bundle = loader.load_data(model_bundle, num_samples=1)
    
    return model_bundle.model, data_bundle.batch, model_bundle.config, model_bundle.device


def tune_thresholds(
    gaussians_full,
    model, tgt_ext, tgt_int, target, h, w, device,
    gt_image, baseline_psnr,
    features, depths,
    quality_budget_pct: float = 0.2,
):
    """
    Sweep SAES v3 multi-level thresholds to find the most aggressive
    settings within the quality budget.
    
    Sweeps: feature_var_threshold (L0), depth_std_threshold (L1),
            saes_threshold (L2). FSGR uses fixed hardware criteria.
    
    Args:
        quality_budget_pct: Max allowed relative PSNR loss in %
    
    Returns:
        (best_saes_threshold, best_fsgr_tolerance, saes_results, fsgr_results)
    """
    from src.model.types import Gaussians
    
    max_loss_db = baseline_psnr * quality_budget_pct / 100.0
    
    def compute_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        return -10 * torch.log10(mse).item()
    
    # ---- Phase 1: Joint SAES v3 sweep (feature_var + depth_std + gauss_threshold) ----
    # Sweep feature_var_threshold (lower = more aggressive L0)
    feat_var_values = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]
    depth_std_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    gauss_thresholds = [0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.95]
    
    print("\n### Threshold Tuning: SAES v3 Multi-Level (Phase 1)")
    print(f"  Quality budget: {quality_budget_pct:.1f}% relative PSNR = {max_loss_db:.4f} dB")
    print(f"  {'FeatVar':>8s}  {'DepthStd':>9s}  {'GaussT':>7s}  {'L0%':>5s}  {'L1%':>5s}  {'L2%':>5s}  "
          f"{'ModPx%':>7s}  {'PSNR':>8s}  {'Loss':>8s}  {'St':>3s}")
    
    best_combo = None
    best_mod_ratio = 0.0
    saes_results = []
    
    for fv in feat_var_values:
        for ds in depth_std_values:
            for gt in gauss_thresholds:
                trial_g = Gaussians(
                    means=gaussians_full.means.clone(),
                    covariances=gaussians_full.covariances.clone(),
                    harmonics=gaussians_full.harmonics.clone(),
                    opacities=gaussians_full.opacities.clone(),
                )
                _, stats, _ = apply_progressive_saes(
                    trial_g, h, w, CONFIG.tile_size, gpp=1,
                    threshold=gt, feature_var_threshold=fv,
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
                l2r = stats.get('level2_ratio', 0.0)
                
                saes_results.append({
                    'feat_var': fv, 'depth_std': ds, 'gauss_thresh': gt,
                    'psnr': psnr, 'loss_db': loss_db, 'loss_pct': loss_pct,
                    'within_budget': within_budget, 'mod_ratio': mod_ratio,
                    'l0_ratio': l0r, 'l1_ratio': l1r, 'l2_ratio': l2r,
                })
                
                st = "OK" if within_budget else "X"
                # Only print interesting combos (high modification OR within budget)
                if within_budget and mod_ratio > 0.2:
                    print(f"  {fv:>8.3f}  {ds:>9.3f}  {gt:>7.3f}  "
                          f"{l0r*100:>5.1f}  {l1r*100:>5.1f}  {l2r*100:>5.1f}  "
                          f"{mod_ratio*100:>7.1f}  {psnr:>8.4f}  {loss_db:>+8.4f}  {st:>3s}")
                
                if within_budget and mod_ratio > best_mod_ratio:
                    best_mod_ratio = mod_ratio
                    best_combo = (fv, ds, gt, psnr, loss_db, mod_ratio)
    
    if best_combo:
        best_fv, best_ds, best_gt, best_psnr, best_loss, best_mod = best_combo
        print(f"\n  ** Best SAES v3: feat_var={best_fv}, depth_std={best_ds}, "
              f"gauss_thresh={best_gt}")
        print(f"     -> {best_mod*100:.1f}% pixels modified, "
              f"PSNR={best_psnr:.4f} dB, loss={best_loss:+.4f} dB")
    else:
        best_fv = 0.015
        best_ds = 0.03
        best_gt = 0.985
        print(f"\n  ** No combo within budget, using defaults")
    
    # ---- Phase 2: FSGR reuse rate (realistic model) ----
    # In the realistic model, FSGR reuse criteria are hardware-fixed (hamming, spatial, confidence).
    # We report the reuse rate for reference — FSGR now has quality impact.
    print(f"\n### FSGR Reuse Rate (Realistic ASIC Model)")
    print(f"  (FSGR uses cached Gaussians → has quality impact)")
    print(f"  Reuse criteria: hamming≤{CONFIG.fsgr_reuse_hamming}, "
          f"spatial≤{CONFIG.fsgr_reuse_spatial}, conf>{CONFIG.fsgr_reuse_confidence}")
    
    fsgr_results = []
    N = gaussians_full.means.shape[1]
    
    has_features = (features is not None and
                   not isinstance(features, str) and
                   hasattr(features, 'shape'))
    
    if has_features:
        B_f, V_f, C_f, H_f, W_f = features.shape
        features_up = F.interpolate(
            features[0], size=(h, w), mode='bilinear', align_corners=False
        ).mean(dim=0)
    
    trial_fsgr = FSGRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsgr_cache_size,
        hamming_threshold=CONFIG.fsgr_hamming_threshold,
        reuse_hamming=CONFIG.fsgr_reuse_hamming,
        reuse_spatial=CONFIG.fsgr_reuse_spatial,
        reuse_confidence=CONFIG.fsgr_reuse_confidence,
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
                trial_fsgr.process_pixel(
                    feat, ad, (y, x), pixel_idx,
                    actual_gaussians=gaussians_full, gauss_idx=pixel_idx,
                )
    
    reuse_rate = trial_fsgr.get_reuse_ratio()
    print(f"  Reuse rate (skip S2+S3): {reuse_rate*100:.1f}%")
    print(f"  Cache hit rate: {trial_fsgr.stats['cache_hits']/max(1,trial_fsgr.stats['total_pixels'])*100:.1f}%")
    
    fsgr_results.append({
        'reuse_rate': reuse_rate,
        'hit_rate': trial_fsgr.stats['cache_hits'] / max(1, trial_fsgr.stats['total_pixels']),
    })
    
    best_fsgr_tol = 0.0  # Not applicable in realistic model
    
    print(f"\n  Best SAES v3: feat_var={best_fv}, depth_std={best_ds}, gauss_thresh={best_gt}")
    
    # Apply the multi-level thresholds to CONFIG
    CONFIG.feature_var_threshold = best_fv
    CONFIG.depth_std_threshold = best_ds
    
    return best_gt, best_fsgr_tol, saes_results, fsgr_results


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
    parser.add_argument('--no-gaussian', action='store_true',
                        help='Disable SCARF gaussian generation HW simulator (GGU), use original GPU')
    
    # Performance optimization options
    parser.add_argument('--no-saes', action='store_true',
                        help='Disable SAES early-stopping optimization')
    parser.add_argument('--no-fsgr', action='store_true',
                        help='Disable FSGR depth reuse optimization')
    
    # Ablation experiment
    parser.add_argument('--ablation', action='store_true',
                        help='Run full ablation: GPU / ASIC / ASIC+FSGR / ASIC+SAES / ASIC+FSGR+SAES. '
                             'Forces both SAES and FSGR to run regardless of --no-saes/--no-fsgr.')
    parser.add_argument('--tune-thresholds', action='store_true',
                        help='Sweep SAES/FSGR thresholds to find optimal settings '
                             'within 0.1%% relative PSNR quality budget.')
    
    # Other options
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run baseline, skip SCARF pipeline')
    parser.add_argument('--freq', type=int, default=1000,
                        help='SCARF ASIC clock frequency in MHz (default: 1000 = 1 GHz)')
    args = parser.parse_args()
    
    # Override global SCARF frequency from CLI
    global SCARF_FREQ_MHZ
    SCARF_FREQ_MHZ = args.freq
    
    # --ablation implies both SAES and FSGR must run
    if args.ablation:
        if args.no_saes:
            print("[ablation] Overriding --no-saes: SAES will run for ablation data")
            args.no_saes = False
        if args.no_fsgr:
            print("[ablation] Overriding --no-fsgr: FSGR will run for ablation data")
            args.no_fsgr = False
    
    # Initialize config
    CONFIG = SCARFConfig(model_type=args.model)
    
    # Create adapter and apply model-specific overrides
    adapter = create_adapter(args.model)
    CONFIG.apply_adapter_overrides(adapter)
    
    # Set model-specific num_depth_candidates
    # Transplat and DepthSplat use 128 depth candidates; MVSplat uses 32
    if args.model in ['transplat', 'depthsplat']:
        CONFIG.num_depth_candidates = 128
    else:
        CONFIG.num_depth_candidates = 32
    
    print("=" * 70)
    print(f"SCARF Demo - Hardware Simulator for 3DGS Encoders")
    print(f"Model: {args.model}")
    print("=" * 70)
    print()
    print(f"[Config] SAES v2:")
    print(f"  tile={CONFIG.tile_size}, threshold={CONFIG.saes_threshold}")
    print(f"  interpolation-based (bilinear from 4 probe Gaussians)")
    print(f"[Config] FSGR (realistic ASIC):")
    print(f"  cache_size={CONFIG.fsgr_cache_size}, hamming_threshold={CONFIG.fsgr_hamming_threshold}")
    print(f"  reuse_hamming={CONFIG.fsgr_reuse_hamming}, reuse_spatial={CONFIG.fsgr_reuse_spatial}")
    print(f"  reuse_confidence={CONFIG.fsgr_reuse_confidence} (cache hit → use cached Gaussians)")
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
    model, batch, cfg, device = load_model_and_data(args.model)
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
    # Step 3: Run BASELINE (full Transplat encoder)
    # --------------------------------------------------------
    print()
    print("[3/6] Running BASELINE (original Transplat)...")
    
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
    # GGU stage controlled by --no-gaussian; S1/S2 always use original GPU
    # When a stage is disabled, use original GPU computation instead of HW simulator
    # --------------------------------------------------------
    print()
    print("[4/6] Running SCARF Pipeline...")
    
    t0 = time.time()
    
    # Initialize cycle counters
    feature_sim_cycles = 0
    depth_sim_cycles = 0
    dp_core_cycles = 0   # DP core: cost_volume + unet + depth_head + regression
    cost_volume_cycles = 0  # cost_volume portion of dp_core (for bandwidth modeling)
    gauss_gen_cycles = 0  # Gaussian Gen: refine_unet + to_gaussians (full-res)
    
    # Initialize FSGR (Feature-Similarity Gaussian Reuse) — realistic ASIC model
    fsgr = FSGRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsgr_cache_size,
        hamming_threshold=CONFIG.fsgr_hamming_threshold,
        reuse_hamming=CONFIG.fsgr_reuse_hamming,
        reuse_spatial=CONFIG.fsgr_reuse_spatial,
        reuse_confidence=CONFIG.fsgr_reuse_confidence,
        num_depth_candidates=CONFIG.num_depth_candidates,
    )
    
    # ============================================================
    # STAGE 1: Feature Extraction
    # ============================================================
    print("  [Stage 1] Feature Extraction...")
    
    # Pipeline variables for Stage 1 output
    pipeline_features = None
    pipeline_cnn_features = None
    
    # Use original GPU for feature extraction
    print(f"    Mode: Original GPU")
    
    with torch.no_grad():
        if args.model == 'depthsplat':
            # DepthSplat: No separate backbone, features are extracted inside depth_predictor
            # We'll handle this in Stage 2 - just mark features as needing extraction
            pipeline_features = 'depthsplat_integrated'  # Marker for Stage 2
            pipeline_cnn_features = None
            print(f"    ✓ Features: (integrated with depth predictor)")
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
            print(f"    ✓ Features: {pipeline_features.shape if pipeline_features is not None else 'None'}")
    
    # ============================================================
    # STAGE 2: Depth Prediction
    # ============================================================
    print("  [Stage 2] Depth Prediction...")
    
    # Pipeline variables for Stage 2 output
    pipeline_depths = None
    pipeline_densities = None
    pipeline_raw_gaussians = None
    
    # Prepare common inputs for depth prediction
    near = context.get('near', target.get('near', torch.tensor([0.5], device=device)))
    far = context.get('far', target.get('far', torch.tensor([100.0], device=device)))
    if near.dim() == 0:
        near = near.unsqueeze(0)
    if far.dim() == 0:
        far = far.unsqueeze(0)
    
    # For Transplat: compute da_depth and dino_feature (needed for both HW and GPU modes)
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
    
    # Use original GPU for depth prediction
    print(f"    Mode: Original GPU")
    
    if pipeline_features is not None and not isinstance(pipeline_features, str) and hasattr(model.encoder, 'depth_predictor'):
        extra_info = {'images': rearrange(context['image'], 'b v c h w -> (v b) c h w')}
        
        with torch.no_grad():
            if args.model == 'transplat':
                pipeline_depths, pipeline_densities, pipeline_raw_gaussians = model.encoder.depth_predictor(
                    pipeline_features,  # Input from Stage 1
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
            elif args.model == 'depthsplat':
                # DepthSplat: run full encoder and capture intermediate outputs via hooks
                captured = {}
                hooks = []
                
                def hook_depth_predictor(module, inputs, outputs):
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        captured['depths'] = outputs[0].detach()
                        captured['densities'] = outputs[1].detach()
                        captured['raw_gaussians'] = outputs[2].detach()
                    elif isinstance(outputs, dict):
                        captured['depths'] = outputs.get('depths', outputs.get('depth'))
                        if captured['depths'] is not None:
                            captured['depths'] = captured['depths'].detach()
                
                def hook_gaussian_head(module, inputs, outputs):
                    captured['gaussian_head_output'] = outputs.detach()
                
                if hasattr(model.encoder, 'depth_predictor'):
                    hooks.append(model.encoder.depth_predictor.register_forward_hook(hook_depth_predictor))
                if hasattr(model.encoder, 'gaussian_head'):
                    hooks.append(model.encoder.gaussian_head.register_forward_hook(hook_gaussian_head))
                
                encoder_output = model.encoder(context, False)
                
                for hook in hooks:
                    hook.remove()
                
                pipeline_depths = captured.get('depths')
                pipeline_densities = captured.get('densities')
                pipeline_raw_gaussians = captured.get('raw_gaussians')
                if 'gaussian_head_output' in captured:
                    pipeline_raw_gaussians = captured['gaussian_head_output']
        
        print(f"    ✓ Depths: {pipeline_depths.shape if pipeline_depths is not None else 'None'}")
        
        if pipeline_depths is not None:
            d_flat = pipeline_depths.reshape(-1)
            print(f"      Depth stats (GPU): min={d_flat.min():.4f}, max={d_flat.max():.4f}, mean={d_flat.mean():.4f}")
        if pipeline_raw_gaussians is not None:
            rg = pipeline_raw_gaussians.reshape(-1)
            print(f"      raw_gaussians stats (GPU): min={rg.min():.4f}, max={rg.max():.4f}, mean={rg.mean():.4f}")
        if pipeline_densities is not None:
            pd = pipeline_densities.reshape(-1)
            print(f"      densities stats (GPU): min={pd.min():.4f}, max={pd.max():.4f}, mean={pd.mean():.4f}")
    
    # ============================================================
    # STAGE 3: Gaussian Generation
    # ============================================================
    print("  [Stage 3] Gaussian Generation...")
    
    # Special case: if ALL hardware simulators are disabled, use baseline directly
    # This avoids error accumulation from running components separately
    all_hw_disabled = args.no_gaussian
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
                # Transplat/MVSplat
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
                    # Transplat/MVSplat format: [B, V, H*W, C]
                    raw_gaussians_parsed = rearrange(pipeline_raw_gaussians, "b v r (srf c) -> b v r srf c", srf=num_surfaces)
                    
                    if args.model == 'depthsplat':
                        # DepthSplat format but in [B, V, H*W, C] format (unlikely but handle it)
                        opacities_raw = raw_gaussians_parsed[..., :1].sigmoid()
                        opacities = opacities_raw / gpp
                        offset_xy = raw_gaussians_parsed[..., 1:3].sigmoid()
                        raw_gaussians_for_ga = raw_gaussians_parsed[..., 3:]
                    else:
                        # Transplat/MVSplat format: offset_xy(2) + scales(3) + rotations(4) + sh(75) = 84
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
                    # Transplat/MVSplat - no input_images
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
    
    # Store pipeline features for FSGR
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
        
        best_saes_th, best_fsgr_tol, _, _ = tune_thresholds(
            scarf_gaussians_full, model, tgt_ext, tgt_int, target, h, w, device,
            gt_image_tune, baseline_psnr_tune,
            features=pipeline_features, depths=pipeline_depths,
        )
        
        # Apply tuned thresholds
        CONFIG.saes_threshold = best_saes_th
        print(f"\n  Applying tuned thresholds: SAES={best_saes_th}")
        
        # Re-create FSGR with tuned config
        fsgr = FSGRSimulator(
            feature_dim=CONFIG.feature_dim,
            cache_size=CONFIG.fsgr_cache_size,
            hamming_threshold=CONFIG.fsgr_hamming_threshold,
            reuse_hamming=CONFIG.fsgr_reuse_hamming,
            reuse_spatial=CONFIG.fsgr_reuse_spatial,
            reuse_confidence=CONFIG.fsgr_reuse_confidence,
            num_depth_candidates=CONFIG.num_depth_candidates,
        )
        # Reset savings tracker
        savings = SavingsTracker()
        print("=" * 70)
    
    # --------------------------------------------------------
    # Step 4b-4d: Real Ablation (SAES v3 multi-level + FSGR + 4-config render)
    # --------------------------------------------------------
    from src.model.types import Gaussians
    N = scarf_gaussians_full.means.shape[1]
    
    # Clone original Gaussians BEFORE any modifications (needed for ablation configs)
    orig_means = scarf_gaussians_full.means.clone()
    orig_covs = scarf_gaussians_full.covariances.clone()
    orig_harmo = scarf_gaussians_full.harmonics.clone()
    orig_opacs = scarf_gaussians_full.opacities.clone()
    
    # ---- Step 4b: SAES v3 (multi-level: feature + depth + Gaussian) ----
    if args.no_saes:
        print("  [4b] Progressive SAES v3: SKIPPED (--no-saes)")
        modified_mask = torch.zeros(N, dtype=torch.bool, device=device)
        saes_stats = {
            'total_tiles_processed': 0, 'early_stop_phase1': 0,
            'early_stop_phase2': 0, 'full_processed': 0,
            'early_stop_ratio': 0.0, 'pixels_interpolated': 0,
            'pixels_original': N, 'interpolation_ratio': 0.0,
            'level0_tiles': 0, 'level1_tiles': 0, 'level2_tiles': 0,
            'full_tiles': 0, 'level0_pixels': 0, 'level1_pixels': 0,
            'level2_pixels': 0, 'total_modified_pixels': 0,
            'level0_ratio': 0.0, 'level1_ratio': 0.0, 'level2_ratio': 0.0,
            'full_ratio': 1.0, 'modification_ratio': 0.0,
        }
        all_pixels = [(y, x, y * w + x) for y in range(h) for x in range(w)]
    else:
        print("  [4b] Progressive SAES v3 (multi-level: feature + depth + Gaussian)...")
        
        # SAES works on a clone so we preserve originals for ablation
        saes_gaussians = Gaussians(
            means=scarf_gaussians_full.means.clone(),
            covariances=scarf_gaussians_full.covariances.clone(),
            harmonics=scarf_gaussians_full.harmonics.clone(),
            opacities=scarf_gaussians_full.opacities.clone(),
        )
        
        modified_mask, saes_stats, continue_pixels = apply_progressive_saes(
            saes_gaussians, h, w, CONFIG.tile_size, gpp=1,
            threshold=CONFIG.saes_threshold,
            feature_var_threshold=CONFIG.feature_var_threshold,
            depth_std_threshold=CONFIG.depth_std_threshold,
            features=features,
            depths=depths,
            cross_check_threshold=CONFIG.saes_cross_check,
        )
        
        # Realistic ASIC model: NO validate-after-compute.
        # In a real ASIC, non-probe pixels are never computed, so there's nothing
        # to validate against. The SAES classification decisions (L0/L1/L2) are
        # based on information available at decision time (S1 features, probe depths,
        # probe Gaussians). The interpolation quality depends on how conservative
        # the classification thresholds are.
        
        # All pixels for FSGR (runs on ALL pixels for its own ablation config)
        all_pixels = [(y, x, y * w + x) for y in range(h) for x in range(w)]
        
        modified_pixels = int(modified_mask.sum().item())
        total_tiles = saes_stats.get('total_tiles_processed', 0)
        print(f"  ✓ Tiles processed: {total_tiles}")
        l0_t = saes_stats.get('level0_tiles', 0)
        l1_t = saes_stats.get('level1_tiles', 0)
        l2_t = saes_stats.get('level2_tiles', 0)
        f_t = saes_stats.get('full_tiles', 0)
        print(f"    Level 0 (feature-uniform):  {l0_t} ({l0_t/max(1,total_tiles)*100:.1f}%)")
        print(f"    Level 1 (depth-uniform):    {l1_t} ({l1_t/max(1,total_tiles)*100:.1f}%)")
        print(f"    Level 2 (Gaussian-similar): {l2_t} ({l2_t/max(1,total_tiles)*100:.1f}%)")
        print(f"    Full (no skip):             {f_t} ({f_t/max(1,total_tiles)*100:.1f}%)")
        print(f"  ✓ Modified pixels (direct interpolation): {modified_pixels:,}/{h*w:,} "
              f"({modified_pixels/(h*w)*100:.1f}%)")
        print(f"    (Realistic: no post-hoc validation, thresholds ensure quality)")
    
    # Record SAES v3 savings for cycle model
    # Use h*w (pixel positions) as base, not N (total Gaussians incl. surfaces)
    # because ASIC processes per pixel position - skipping a position skips all surfaces
    savings.record_saes(total_pixels=h*w, saes_stats=saes_stats)
    
    # ---- Step 4c: FSGR (Feature-Similarity Gaussian Reuse) — Realistic ASIC ----
    if args.no_fsgr:
        print("  [4c] FSGR: SKIPPED (--no-fsgr)")
    else:
        print("  [4c] FSGR (Realistic ASIC: cache hit → use cached Gaussians)...")
        
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
                
                # FSGR processing (realistic: reuse decision before compute)
                path, num_searches, output_depth = fsgr.process_pixel(
                    feat, actual_depth, (y, x), pixel_idx,
                    actual_gaussians=scarf_gaussians_full,
                    gauss_idx=pixel_idx,
                )
                
                # Record path for savings tracking
                savings.record_fsgr_pixel(path, reused=(pixel_idx in fsgr.reuse_data))
        
        fsgr_stats = fsgr.get_summary()
        if fsgr_stats['total_pixels'] > 0:
            print(f"    Pixels processed: {fsgr_stats['total_pixels']:,}")
            print(f"    Cache hit rate: {fsgr_stats['hit_rate']*100:.1f}%")
            print(f"    Guided (narrowed): {fsgr_stats.get('guided', 0):,} "
                  f"({fsgr_stats.get('guided_rate', 0)*100:.1f}%)")
            print(f"    Depth inconsistent: {fsgr_stats.get('depth_inconsistent', 0):,}")
            print(f"    Not guided: {fsgr_stats.get('hit_no_guide', 0):,}")
            print(f"    Full compute (miss): {fsgr_stats['full_compute']:,}")
            print(f"    Criteria: hamming≤{fsgr.reuse_hamming}, conf>{fsgr.reuse_confidence:.2f}")
    
    # Record GGU cycles
    cycle_counter.add_ggu(N)
    
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
    
    # Config 2: +FSGR only (narrowed search: most pixels zero quality impact)
    if not args.no_fsgr and len(fsgr.reuse_data) > 0:
        out_window_count = sum(1 for v in fsgr.reuse_data.values() if not v.get('in_window', True))
        print(f"  [2/4] Rendering: +FSGR only ({len(fsgr.reuse_data):,} guided, "
              f"{out_window_count} outside window)...")
        fsgr_means = orig_means.clone()
        for pixel_idx, reuse_info in fsgr.reuse_data.items():
            # Narrowed search: only modify means for pixels where actual depth
            # was outside the narrowed window (rare, depth consistency prevents most)
            depth_ratio = reuse_info['depth_ratio']
            if abs(depth_ratio - 1.0) > 1e-6:  # Only modify if ratio != 1.0
                fsgr_means[0, pixel_idx] = orig_means[0, pixel_idx] * depth_ratio
        g_fsgr = Gaussians(means=fsgr_means, covariances=orig_covs.clone(),
                           harmonics=orig_harmo.clone(), opacities=orig_opacs.clone())
        ablation_renders['asic_fsgr'] = _render(g_fsgr)
        depth_err = fsgr.get_depth_error_stats()
        if depth_err['count'] > 0:
            print(f"    Out-of-window depth error: mean={depth_err['mean']:.4f}, "
                  f"p95={depth_err['p95']:.4f}, max={depth_err['max']:.4f}")
        else:
            print(f"    All guided pixels had actual depth within window → zero quality impact")
    else:
        print("  [2/4] Rendering: +FSGR only (no guided / disabled)...")
        ablation_renders['asic_fsgr'] = ablation_renders['asic']
    
    # Config 3: +SAES only (SAES-interpolated appearance, no post-hoc validation)
    if not args.no_saes:
        print("  [3/4] Rendering: +SAES only (direct interpolation, no oracle validation)...")
        ablation_renders['asic_saes'] = _render(saes_gaussians)
    else:
        ablation_renders['asic_saes'] = ablation_renders['asic']
    
    # Config 4: +FSGR + SAES (SAES interpolation + FSGR depth-only reuse on non-SAES pixels)
    if not args.no_saes or (not args.no_fsgr and len(fsgr.reuse_data) > 0):
        print("  [4/4] Rendering: +FSGR+SAES (combined realistic)...")
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
        
        # Apply FSGR narrowed search means modification on non-SAES pixels only
        if not args.no_fsgr:
            fsgr_on_non_saes = 0
            fsgr_modified = 0
            for pixel_idx, reuse_info in fsgr.reuse_data.items():
                if not modified_mask[pixel_idx]:  # Not modified by SAES
                    fsgr_on_non_saes += 1
                    depth_ratio = reuse_info['depth_ratio']
                    if abs(depth_ratio - 1.0) > 1e-6:
                        combo_means[0, pixel_idx] = orig_means[0, pixel_idx] * depth_ratio
                        fsgr_modified += 1
            print(f"    FSGR guided {fsgr_on_non_saes:,} non-SAES pixels "
                  f"({fsgr_modified} means-modified)")
        
        g_combo = Gaussians(means=combo_means, covariances=combo_covs,
                            harmonics=combo_harmo, opacities=combo_opacs)
        ablation_renders['asic_fsgr_saes'] = _render(g_combo)
    else:
        ablation_renders['asic_fsgr_saes'] = ablation_renders['asic']
    
    # Use the best config (FSGR+SAES) as the "SCARF image" for backward compatibility
    scarf_image = ablation_renders['asic_fsgr_saes']
    scarf_time = time.time() - t0
    
    # Gaussian stats
    fsgr_reused = fsgr.stats['total_reuse'] if not args.no_fsgr else 0
    gaussian_stats = {
        'gaussians_baseline': N,
        'gaussians_output': N,
        'pixels_modified': saes_stats.get('total_modified_pixels', 0),
        'fsgr_reused': fsgr_reused,
    }
    
    print(f"  ✓ 4-config rendering complete ({scarf_time:.2f}s total)")
    print(f"  ✓ SAES modified {gaussian_stats['pixels_modified']:,} pixels (interpolation)")
    print(f"  ✓ FSGR guided {fsgr_reused:,} pixels (narrowed search, 32/{CONFIG.num_depth_candidates} candidates)")
    
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
    
    # Use FSGR+SAES as the "SCARF" result
    scarf_psnr = ablation_quality['asic_fsgr_saes']['psnr']
    scarf_ssim = ablation_quality['asic_fsgr_saes']['ssim']
    
    # Check quality budget (compare optimized configs vs no-opt, excluding no-opt itself)
    opt_configs = {k: v for k, v in ablation_quality.items() if k != 'asic'}
    worst_loss = max(q['loss_pct'] for q in opt_configs.values()) if opt_configs else 0
    if worst_loss <= 0.2:
        print(f"  ✓ All optimized configs within 0.2% quality budget (worst: {worst_loss:.4f}%)")
    else:
        print(f"  ⚠ Quality budget exceeded: worst config {worst_loss:.4f}% > 0.2%")
    
    # Save outputs
    output_dir = SCARF_ROOT / 'outputs' / 'demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from torchvision.utils import save_image
    save_image(gt_image, output_dir / 'gt_00.png')
    save_image(baseline_image, output_dir / 'baseline_00.png')
    save_image(scarf_image, output_dir / 'scarf_00.png')
    for cfg_key, cfg_image in ablation_renders.items():
        save_image(cfg_image, output_dir / f'ablation_{cfg_key}.png')
    
    # Get statistics
    ggu_stats = cycle_counter.get_summary()
    fsgr_summary = fsgr.get_summary()
    
    # Base cycle counts (no SAES/FSGR savings)
    base_ggu_cycles = ggu_stats['ggu_cycles']
    
    # ================================================================
    # Fallback: Reference cycle counts from Transplat 256×256 HW profile
    # When feature_extractor / depth_predictor simulators are not available,
    # use architecture-derived reference values (deterministic for given model).
    # These values were profiled from the 32×32 base hardware simulators
    # and are scaled by compute_ablation's differentiated HW scaling.
    # ================================================================
    if feature_sim_cycles == 0:
        # TransplatFeatureExtractor: CNN backbone + ViT transformer layers
        feature_sim_cycles = 155_273_000
        print(f"    [Fallback] Using reference S1 feature cycles: {feature_sim_cycles:,}")
    
    if dp_core_cycles == 0:
        # TransplatDepthPredictorSim: cost_volume + unet_refinement + depth_head + regression
        # cost_volume (128 candidates, bilinear warping): ~193M cycles (memory-bound)
        # unet + depth_head + regression: ~116M cycles (compute-bound)
        dp_core_cycles = 309_154_000
        cost_volume_cycles = 193_338_000
        depth_sim_cycles = dp_core_cycles  # total S2 for legacy compatibility
        print(f"    [Fallback] Using reference S2 depth cycles: {dp_core_cycles:,} (cv: {cost_volume_cycles:,})")
    
    if gauss_gen_cycles == 0:
        # Gaussian generation head: refine_unet + to_gaussians (full-res spatial processing)
        gauss_gen_cycles = 304_914_000
        print(f"    [Fallback] Using reference S3 gauss gen cycles: {gauss_gen_cycles:,}")
    
    # Compute ablation table (with real savings from SAES/FSGR)
    ablation = savings.compute_ablation(
        feature_cycles=feature_sim_cycles,
        depth_cycles=depth_sim_cycles,
        ggu_cycles=base_ggu_cycles,
        dp_core_cycles=dp_core_cycles,
        gauss_gen_cycles=gauss_gen_cycles,
        cost_volume_cycles=cost_volume_cycles,
        ablation_quality=ablation_quality,
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
        'asic_fsgr': 'ASIC + FSGR',
        'asic_saes': 'ASIC + SAES v3',
        'asic_fsgr_saes': 'ASIC + SAES+FSGR',
    }
    for cfg_key in ['asic', 'asic_fsgr', 'asic_saes', 'asic_fsgr_saes']:
        q = ablation_quality.get(cfg_key, {})
        label = cfg_labels.get(cfg_key, cfg_key)
        print(f"  {label:<24s}: PSNR={q.get('psnr', 0):.4f} dB, SSIM={q.get('ssim', 0):.6f}, "
              f"loss={q.get('loss_db', 0):+.4f} dB ({q.get('loss_pct', 0):.4f}%)")
    print()
    print("### SAES v3 Results (Multi-Level)")
    print(f"  Total tiles:    {saes_stats.get('total_tiles_processed', 0)}")
    print(f"  Level 0 (feat): {saes_stats.get('level0_tiles', 0)} tiles "
          f"({saes_stats.get('level0_ratio', 0)*100:.1f}%), {savings.level0_pixels:,} px replicated")
    print(f"  Level 1 (depth):{saes_stats.get('level1_tiles', 0)} tiles "
          f"({saes_stats.get('level1_ratio', 0)*100:.1f}%), {savings.level1_pixels:,} px interpolated")
    print(f"  Level 2 (gauss):{saes_stats.get('level2_tiles', 0)} tiles "
          f"({saes_stats.get('level2_ratio', 0)*100:.1f}%), {savings.level2_pixels:,} px interpolated")
    print(f"  Full (no skip): {saes_stats.get('full_tiles', 0)} tiles "
          f"({saes_stats.get('full_ratio', 0)*100:.1f}%)")
    print(f"  Total modified: {savings.saes_interpolated_pixels:,}/{savings.total_pixels:,} "
          f"({savings.saes_interpolation_ratio()*100:.1f}%)")
    print(f"  All Gaussians KEPT (modified, not removed)")
    print()
    print("### FSGR Statistics (Narrowed Depth Search ASIC Model)")
    fsgr_summary = fsgr.get_summary()
    if fsgr_summary['total_pixels'] > 0:
        print(f"  Pixels processed:     {fsgr_summary['total_pixels']:,}")
        print(f"  Cache hit rate:       {fsgr_summary['hit_rate']*100:.1f}%")
        print(f"  Guided (32 cand.):    {fsgr_summary.get('guided', 0):,} "
              f"({fsgr_summary['guided_rate']*100:.1f}%)")
        print(f"    In window:          {fsgr_summary.get('guided_in_window', 0):,} "
              f"({fsgr_summary.get('in_window_rate', 0)*100:.1f}% → zero quality impact)")
        print(f"    Out of window:      {fsgr_summary.get('guided_out_window', 0):,} "
              f"(means modified)")
        print(f"  Depth inconsistent:   {fsgr_summary.get('depth_inconsistent', 0):,} "
              f"(→ full 128-candidate search)")
        print(f"  Not guided (criteria):{fsgr_summary.get('hit_no_guide', 0):,} "
              f"({fsgr_summary.get('hit_no_guide_rate', 0)*100:.1f}%)")
        print(f"  Full compute (miss):  {fsgr_summary['full_compute']:,} "
              f"({fsgr_summary['full_compute_rate']*100:.1f}%)")
        fsgr_save_pct = pipe_info.get('fsgr_s2_save_per_pixel', 0.42) * 100
        print(f"  S2 saving per guided: {fsgr_save_pct:.1f}% "
              f"(cost_volume only, 32/{CONFIG.num_depth_candidates} candidates)")
        depth_err = fsgr_summary.get('depth_error', {})
        if depth_err.get('count', 0) > 0:
            print(f"  Out-window depth err: mean={depth_err['mean']:.4f}, max={depth_err['max']:.4f}")
        print(f"  Criteria:             hamming≤{fsgr.reuse_hamming}, "
              f"conf>{fsgr.reuse_confidence:.2f}, "
              f"depth_consist≤{fsgr.depth_consistency_threshold:.0%}")
    else:
        print(f"  (No FSGR pixels processed)")
    
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
    cv_frac = pipe_info.get('cv_frac_scaled', 0)
    print(f"  S2 cost_volume frac:{cv_frac*100:5.1f}%  (after scaling; memory-bound portion)")
    print(f"  S1 FE overlap:      x{pipe_info.get('PIPE_FE', 0.92):.2f}  "
          f"(ConvEngine || GEMM in ViT, double buffering)")
    print(f"  S2 DP overlap:      x{pipe_info.get('PIPE_DP', 0.82):.2f}  "
          f"(BilinearUnit || VectorALU || ConvEngine tile pipeline)")
    print(f"  S3 GaussNN overlap: x{pipe_info.get('PIPE_GG_NN', 0.95):.2f}  "
          f"(refine_unet → to_gauss output buf || weight prefetch)")
    print(f"  GGU Post:           hidden (dedicated PEs overlap with ConvEngine)")
    print(f"  SAES v3 multi-level:")
    print(f"    L0 (feature):     {pipe_info.get('saes_l0_ratio', 0)*100:.1f}% tiles (4-probe, saves 75% S2+S3)")
    print(f"    L1 (depth):       {pipe_info.get('saes_l1_ratio', 0)*100:.1f}% tiles (saves 75% S2+S3)")
    print(f"    L2 (Gaussian):    {pipe_info.get('saes_l2_ratio', 0)*100:.1f}% tiles (saves 75% S3)")
    print(f"    Combined S2 save: {pipe_info.get('saes_s2_saving', 0)*100:.1f}%")
    print(f"    Combined S3 save: {pipe_info.get('saes_s3_saving', 0)*100:.1f}%")
    fsgr_save_pp = pipe_info.get('fsgr_s2_save_per_pixel', 0.42)
    print(f"  FSGR guided:        {pipe_info.get('fsgr_reuse_ratio', 0)*100:.1f}% "
          f"(narrowed S2: 32/{CONFIG.num_depth_candidates} candidates, "
          f"saves {fsgr_save_pp*100:.1f}%/pixel, cost_volume only)")
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
    
    # [5] ASIC + FSGR
    _print_asic_config(f"[{idx}] SCARF ASIC + FSGR", ablation['asic_fsgr'], ggu_pe)
    idx += 1
    
    # [6] ASIC + SAES v3
    _print_asic_config(f"[{idx}] SCARF ASIC + SAES v3", ablation['asic_saes'], ggu_pe)
    idx += 1
    
    # [7] ASIC + SAES v3 + FSGR
    asic_best_ms = _print_asic_config(f"[{idx}] SCARF ASIC + SAES v3 + FSGR", ablation['asic_fsgr_saes'], ggu_pe)
    
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
    asic_best_time = _eff_ms('asic_fsgr_saes')
    
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
        ('asic_fsgr',      f"  + FSGR"),
        ('asic_saes',      f"  + SAES v3"),
        ('asic_fsgr_saes', f"  + SAES v3+FSGR << best"),
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
    best_q = ablation_quality.get('asic_fsgr_saes', {})
    print(f"  Best config quality:  PSNR={best_q.get('psnr', 0):.4f} dB, "
          f"loss={best_q.get('loss_db', 0):+.4f} dB ({best_q.get('loss_pct', 0):.4f}%)")
    print(f"  SAES v3 total:        {savings.saes_interpolation_ratio()*100:.1f}% pixels modified "
          f"(L0={savings.level0_ratio()*100:.1f}%, L1={savings.level1_ratio()*100:.1f}%, "
          f"L2={savings.level2_ratio()*100:.1f}%)")
    print(f"  FSGR guided:          {savings.fsgr_validated_saving_ratio()*100:.1f}% "
          f"({savings.fsgr_validated:,} pixels narrowed S2 search)")
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
    total_eff = ablation['asic_fsgr_saes'].get('eff_total', 0)
    eff_fe = ablation['asic_fsgr_saes'].get('eff_feature', 0)
    eff_dp = ablation['asic_fsgr_saes'].get('eff_dp_core', 0)
    eff_gg = ablation['asic_fsgr_saes'].get('eff_gauss_gen', 0)
    
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
    a_misc_logic = 0.3 + 0.15  # activation/norm + FSGR/SAES
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
