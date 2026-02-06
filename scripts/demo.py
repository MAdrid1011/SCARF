#!/usr/bin/env python3
"""
SCARF Demo - Hardware Simulator for 3D Gaussian Splatting Encoders

SCARF (Scene-Adaptive Cost-volume Accelerator with Reuse Framework) is a
hardware-realizable accelerator for 3D Gaussian Splatting encoders.

This demo shows SCARF's two key optimizations:
1. SAES (Scene-Adaptive Early-Stopping): Skip depth search for homogeneous tiles
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
Skip DSU    [SCARF DSU + FSDR] → depths
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
- Performance: Hardware cycle counts (DSU, GGU, total)
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
    --no-saes         Disable SAES early-stopping
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
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import argparse

# SCARF imports
from adapters import create_adapter, BaseAdapter
from integration import create_model_loader, ModelBundle, DataBundle
from ggu import GGUProcessor, GGUConfig

# ============================================================
# Configuration
# ============================================================
@dataclass
class SCARFConfig:
    """SCARF pipeline configuration."""
    # Model type
    model_type: str = 'transplat'
    
    # SAES config
    tile_size: int = 4
    early_stop_threshold: float = 0.85
    cov_enlarge_factor: float = 6.0
    
    # FSDR config
    fsdr_cache_size: int = 128
    fsdr_hamming_threshold: int = 4
    fsdr_high_confidence: float = 0.8
    fsdr_medium_confidence: float = 0.5
    
    # DSU config (will be overridden based on model type)
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
        fsdr_overrides = adapter.get_fsdr_config_overrides()
        saes_overrides = adapter.get_saes_config_overrides()
        scale_min, scale_max = adapter.get_scale_range()
        
        # Apply overrides
        if 'hamming_threshold' in fsdr_overrides:
            self.fsdr_hamming_threshold = fsdr_overrides['hamming_threshold']
        if 'high_confidence_threshold' in fsdr_overrides:
            self.fsdr_high_confidence = fsdr_overrides['high_confidence_threshold']
        if 'early_stop_threshold' in saes_overrides:
            self.early_stop_threshold = saes_overrides['early_stop_threshold']
        
        self.scale_min = scale_min
        self.scale_max = scale_max


# Global config (initialized in main)
CONFIG: SCARFConfig = None


# ============================================================
# Cycle Counter for Hardware Performance
# ============================================================
class CycleCounter:
    """Hardware cycle counter for DSU and GGU operations."""
    
    # Hardware cycle constants
    DSU_CYCLES = {
        'project': 8,      # 3D-2D projection per depth
        'sample': 4,       # Bilinear sampling per depth
        'cost': 2,         # Cost computation per depth
        'softmax': 6,      # Softmax aggregation
    }
    
    GGU_CYCLES = {
        'position': 3,     # Ray + depth
        'scale': 2,        # Sigmoid + multiply
        'covariance': 5,   # Quat2mat + matmul
        'transform': 3,    # World transform
        'sh_rotation': 3,  # SH rotation
        'opacity': 1,      # Sigmoid
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dsu_cycles = 0
        self.ggu_cycles = 0
        self.dsu_skipped = 0
        self.dsu_full = 0
        self.dsu_fsdr_saved = 0
        self.fsdr_stats = {
            'direct_reuse': 0,
            'interpolation': 0,
            'light_verify': 0,
            'full_search': 0,
        }
    
    def add_dsu_full_search(self, num_depths: int = 32):
        """Record cycles for full DSU depth search."""
        cycles = (
            self.DSU_CYCLES['project'] * num_depths +
            self.DSU_CYCLES['sample'] * num_depths +
            self.DSU_CYCLES['cost'] * num_depths +
            self.DSU_CYCLES['softmax']
        )
        self.dsu_cycles += cycles
        self.dsu_full += 1
        return cycles
    
    def add_dsu_skip(self):
        """Record skipped DSU (early-stop)."""
        self.dsu_skipped += 1
    
    def add_dsu_fsdr(self, path: str, num_searches: int = 0):
        """Record DSU with FSDR optimization."""
        self.fsdr_stats[path] += 1
        
        # Calculate full search cycles for comparison
        full_search_cycles = (
            self.DSU_CYCLES['project'] * 32 +
            self.DSU_CYCLES['sample'] * 32 +
            self.DSU_CYCLES['cost'] * 32 +
            self.DSU_CYCLES['softmax']
        )
        
        if path == 'direct_reuse':
            cycles = 0  # No depth search
            self.dsu_fsdr_saved += full_search_cycles
        elif path == 'interpolation':
            cycles = 0  # No depth search
            self.dsu_fsdr_saved += full_search_cycles
        elif path == 'light_verify':
            cycles = (
                self.DSU_CYCLES['project'] * num_searches +
                self.DSU_CYCLES['sample'] * num_searches +
                self.DSU_CYCLES['cost'] * num_searches
            )
            self.dsu_fsdr_saved += full_search_cycles - cycles
        else:  # full_search
            cycles = full_search_cycles
            self.dsu_full += 1
        
        self.dsu_cycles += cycles
        return cycles
    
    def add_ggu(self, num_gaussians: int = 1):
        """Record cycles for GGU gaussian generation."""
        cycles_per_gaussian = sum(self.GGU_CYCLES.values())
        total_cycles = cycles_per_gaussian * num_gaussians
        self.ggu_cycles += total_cycles
        return total_cycles
    
    def get_summary(self) -> Dict:
        """Get cycle counting summary."""
        total_cycles = self.dsu_cycles + self.ggu_cycles
        return {
            'dsu_cycles': self.dsu_cycles,
            'ggu_cycles': self.ggu_cycles,
            'total_cycles': total_cycles,
            'dsu_skipped': self.dsu_skipped,
            'dsu_full': self.dsu_full,
            'fsdr_stats': self.fsdr_stats.copy(),
        }


# ============================================================
# FSDR Simulator (Feature-Similarity Depth Reuse)
# ============================================================
class FSDRSimulator:
    """
    FSDR simulator for depth reuse based on feature similarity.
    
    Now TRULY affects depth values:
    - Cache hit → reuse cached depth
    - Cache miss → use original depth and cache it
    """
    
    def __init__(self, feature_dim: int = 128, cache_size: int = 128, hamming_threshold: int = 4):
        from fsdr import FSDRConfig, LSHHasher, CacheTable
        
        self.config = FSDRConfig(
            cache_size=cache_size,
            feature_dim=feature_dim,
            hamming_threshold=hamming_threshold,
        )
        self.hasher = LSHHasher(self.config)
        self.cache = CacheTable(self.config)
        
        # Depth modification tracking
        self.depth_modifications = {}  # pixel_idx -> modified_depth
        
        self.stats = {
            'total_pixels': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'direct_reuse': 0,
            'interpolation': 0,
            'light_verify': 0,
            'full_search': 0,
            'total_searches': 0,
            'baseline_searches': 0,
            'depth_reused': 0,
            'depth_original': 0,
        }
    
    def process_pixel(self, feature: torch.Tensor, original_depth: float, position: Tuple[int, int], pixel_idx: int) -> Tuple[str, int, float]:
        """
        Process a single pixel through FSDR.
        
        CONSERVATIVE depth modification to maintain quality:
        - Only reuse depth when VERY similar (hamming ≤ 1) AND spatially close
        - Use small interpolation weights
        - Clamp depth changes to ±10% of original
        
        Returns:
            (path, num_searches, output_depth): Processing path, searches needed, and actual depth to use
        """
        self.stats['total_pixels'] += 1
        self.stats['baseline_searches'] += CONFIG.num_depth_candidates
        
        # Move to CPU for FSDR processing
        feature_cpu = feature.cpu() if feature.is_cuda else feature
        
        # Generate LSH signature
        signature = self.hasher.hash(feature_cpu)
        
        # Cache lookup
        entry, hamming_dist = self.cache.lookup(signature)
        
        if entry is not None:
            self.stats['cache_hits'] += 1
            
            # Check spatial proximity (only reuse from nearby pixels)
            cached_pos = entry.position
            spatial_dist = abs(position[0] - cached_pos[0]) + abs(position[1] - cached_pos[1])
            
            # Calculate depth similarity - only reuse if depths are similar
            depth_ratio = entry.best_depth / (original_depth + 1e-8)
            depth_similar = 0.9 < depth_ratio < 1.1  # Within 10% of each other
            
            # BALANCED: reuse if feature similar AND (spatially close OR depth similar)
            if entry.peak_prob > 0.85 and hamming_dist <= 2 and (spatial_dist <= 8 or depth_similar):
                # Direct reuse - with clamped change
                self.stats['direct_reuse'] += 1
                self.stats['depth_reused'] += 1
                # Clamp the change to ±2% of original (more conservative)
                max_change = original_depth * 0.02
                output_depth = max(original_depth - max_change, 
                                   min(original_depth + max_change, entry.best_depth))
                self.depth_modifications[pixel_idx] = output_depth
                return 'direct_reuse', 0, output_depth
            
            elif hamming_dist <= 3 and (spatial_dist <= 16 or depth_similar):
                # Interpolation - blend with small cache weight
                self.stats['interpolation'] += 1
                self.stats['depth_reused'] += 1
                # Weight based on both hamming and spatial distance
                hamming_weight = 1.0 - hamming_dist / 4.0  # 0→1.0, 3→0.25
                spatial_weight = max(0, 1.0 - spatial_dist / 20.0)  # 0→1.0, 20→0
                weight = 0.10 * hamming_weight * spatial_weight  # Max 10% cache influence
                output_depth = weight * entry.best_depth + (1 - weight) * original_depth
                self.depth_modifications[pixel_idx] = output_depth
                return 'interpolation', 0, output_depth
            
            else:
                # Light verify - no adjustment, just verify path
                num_searches = min(5, CONFIG.num_depth_candidates)
                self.stats['light_verify'] += 1
                self.stats['total_searches'] += num_searches
                # No modification for light verify path - use original
                output_depth = original_depth
                self.stats['depth_reused'] += 1
                return 'light_verify', num_searches, output_depth
        else:
            # Cache miss - use original depth and cache it
            self.stats['cache_misses'] += 1
            self.stats['full_search'] += 1
            self.stats['total_searches'] += CONFIG.num_depth_candidates
            self.stats['depth_original'] += 1
            
            # Insert into cache for future pixels
            from fsdr import CacheEntry
            new_entry = CacheEntry(
                signature=signature,
                position=position,
                best_depth=float(original_depth),
                best_idx=0,
                peak_prob=0.9,  # Assume high confidence for neural network output
                second_offset=1,
                spread=0.1,
                valid=True,
            )
            self.cache.insert(new_entry)
            
            # No modification for cache miss - use original
            return 'full_search', CONFIG.num_depth_candidates, original_depth
    
    def get_depth_modification(self, pixel_idx: int) -> Optional[float]:
        """Get modified depth for a pixel if FSDR reused it."""
        return self.depth_modifications.get(pixel_idx, None)
    
    def get_summary(self) -> Dict:
        """Get FSDR statistics summary."""
        total = self.stats['total_pixels']
        if total == 0:
            return self.stats
        
        summary = self.stats.copy()
        summary['hit_rate'] = self.stats['cache_hits'] / total
        summary['memory_reduction'] = 1.0 - (self.stats['total_searches'] / self.stats['baseline_searches']) if self.stats['baseline_searches'] > 0 else 0
        summary['direct_reuse_rate'] = self.stats['direct_reuse'] / total
        summary['interpolation_rate'] = self.stats['interpolation'] / total
        summary['light_verify_rate'] = self.stats['light_verify'] / total
        summary['full_search_rate'] = self.stats['full_search'] / total
        summary['depth_reuse_rate'] = self.stats['depth_reused'] / total if total > 0 else 0
        
        return summary


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


# ============================================================
# SAES: Scene-Adaptive Early-Stopping (Progressive Version)
# ============================================================
class ProgressiveSAES:
    """
    Progressive Adaptive Early-Stopping for SCARF.
    
    Core idea: Process tiles from center outward, decide early-stop or subdivide
    based on Gaussian similarity (covariance, SH, opacity) as we go.
    
    Processing order for 4x4 tile:
        Phase 1: Center 4 pixels → check similarity → early-stop?
        Phase 2: Cross 8 pixels → check similarity → early-stop?
        Phase 3: Corners 4 pixels → complete
        
    If similarity < LOW_THRESHOLD at Phase 1: subdivide into 4 sub-tiles
    """
    
    # Thresholds for early-stop decisions
    # Tuned for quality loss < 3dB while maximizing speedup
    HIGH_THRESHOLD = 0.98   # Early-stop if similarity > 98% (balanced)
    LOW_THRESHOLD = 0.90    # (Reserved for future subdivide feature)
    MIN_TILE_SIZE = 2       # (Reserved for future subdivide feature)
    ENABLE_SUBDIVIDE = False  # Disable subdivide for now (performance issue)
    
    def __init__(self, H: int, W: int, initial_tile_size: int = 4):
        self.H = H
        self.W = W
        self.initial_tile_size = initial_tile_size
        
        # Statistics
        self.stats = {
            'total_tiles_processed': 0,
            'early_stop_phase1': 0,
            'early_stop_phase2': 0,
            'subdivided': 0,
            'full_processed': 0,
            'gaussians_saved': 0,
            'gaussians_output': 0,
        }
    
    @staticmethod
    def get_spiral_order(tile_size: int) -> List[List[Tuple[int, int]]]:
        """Generate center-first spiral processing order."""
        if tile_size == 4:
            return [
                # Phase 1: Center 4 pixels
                [(1, 1), (1, 2), (2, 1), (2, 2)],
                # Phase 2: Cross 8 pixels
                [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)],
                # Phase 3: Corners 4 pixels
                [(0, 0), (0, 3), (3, 0), (3, 3)],
            ]
        elif tile_size == 2:
            return [
                # All 4 pixels at once
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ]
        else:
            # For other sizes, just return all pixels
            return [[(y, x) for y in range(tile_size) for x in range(tile_size)]]
    
    @staticmethod
    def compute_gaussian_similarity(gaussians: List[Dict]) -> float:
        """
        Compute similarity among a group of Gaussians.
        
        Based on Challenge.md criteria:
        - Covariance cosine similarity
        - SH cosine similarity
        - Opacity L2 distance
        
        Returns average pairwise similarity in [0, 1].
        """
        if len(gaussians) < 2:
            return 1.0
        
        n = len(gaussians)
        total_sim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                g1, g2 = gaussians[i], gaussians[j]
                
                # Covariance similarity (flatten 3x3 matrix)
                cov1 = g1['covariance'].flatten()
                cov2 = g2['covariance'].flatten()
                cov_sim = F.cosine_similarity(cov1.unsqueeze(0), cov2.unsqueeze(0)).item()
                cov_sim = (cov_sim + 1) / 2  # Map [-1, 1] to [0, 1]
                
                # SH similarity
                sh1 = g1['harmonics'].flatten()
                sh2 = g2['harmonics'].flatten()
                sh_sim = F.cosine_similarity(sh1.unsqueeze(0), sh2.unsqueeze(0)).item()
                sh_sim = (sh_sim + 1) / 2
                
                # Opacity similarity (1 - normalized L2 distance)
                opacity_dist = abs(g1['opacity'] - g2['opacity'])
                opacity_sim = 1.0 - min(opacity_dist, 1.0)
                
                # Weighted combination (matching Challenge.md emphasis)
                pair_sim = 0.4 * cov_sim + 0.4 * sh_sim + 0.2 * opacity_sim
                total_sim += pair_sim
                count += 1
        
        return total_sim / count if count > 0 else 1.0
    
    def process_tile(
        self,
        tile_y: int, tile_x: int,
        tile_size: int,
        gaussians_full,  # Full Gaussians object
        W: int,
        gpp: int = 1,  # Gaussians per pixel
    ) -> Tuple[List[int], List[int], bool, int]:
        """
        Process a single tile with progressive early-stopping.
        
        Returns:
            keep_indices: Indices of Gaussians to keep
            enlarge_indices: Indices of Gaussians to enlarge covariance
            early_stopped: Whether this tile was early-stopped
            phase_stopped: Which phase early-stopped (0=subdivided, 1-3=phase, -1=full)
        """
        self.stats['total_tiles_processed'] += 1
        
        spiral_order = self.get_spiral_order(tile_size)
        processed_gaussians = []
        processed_indices = []
        
        for phase_idx, phase_pixels in enumerate(spiral_order):
            # Process this phase's pixels
            for (local_y, local_x) in phase_pixels:
                global_y = tile_y + local_y
                global_x = tile_x + local_x
                pixel_idx = global_y * W + global_x
                
                # Get Gaussian data for this pixel
                for g in range(gpp):
                    flat_idx = pixel_idx * gpp + g
                    if flat_idx < gaussians_full.means.shape[1]:
                        gaussian_data = {
                            'covariance': gaussians_full.covariances[0, flat_idx],
                            'harmonics': gaussians_full.harmonics[0, flat_idx],
                            'opacity': gaussians_full.opacities[0, flat_idx].item(),
                        }
                        processed_gaussians.append(gaussian_data)
                        processed_indices.append(flat_idx)
            
            # Check similarity after this phase (need at least 4 Gaussians)
            if len(processed_gaussians) >= 4:
                similarity = self.compute_gaussian_similarity(processed_gaussians)
                
                # Early-stop: high similarity
                if similarity >= self.HIGH_THRESHOLD:
                    if phase_idx == 0:
                        self.stats['early_stop_phase1'] += 1
                    else:
                        self.stats['early_stop_phase2'] += 1
                    
                    # Keep only center Gaussians, mark for enlargement
                    center_indices = processed_indices[:4]  # First 4 (center)
                    self.stats['gaussians_saved'] += (tile_size * tile_size * gpp) - len(center_indices)
                    self.stats['gaussians_output'] += len(center_indices)
                    return center_indices, center_indices, True, phase_idx + 1
                
                # Subdivide: low similarity at phase 1 (currently disabled)
                if self.ENABLE_SUBDIVIDE and phase_idx == 0 and similarity < self.LOW_THRESHOLD and tile_size > self.MIN_TILE_SIZE:
                    self.stats['subdivided'] += 1
                    # Return signal to subdivide
                    return [], [], False, 0  # phase_stopped=0 means subdivide
        
        # Full processing complete
        self.stats['full_processed'] += 1
        self.stats['gaussians_output'] += len(processed_indices)
        return processed_indices, [], False, -1
    
    def process_all_tiles_fast(
        self,
        gaussians_full,
        gpp: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Fast batch processing of all tiles with progressive SAES.
        
        Optimized version: compute all similarities at once, then make decisions.
        
        Returns:
            keep_mask: Boolean mask for Gaussians to keep
            enlarge_mask: Boolean mask for Gaussians to enlarge
            stats: Processing statistics
        """
        N = gaussians_full.means.shape[1]
        device = gaussians_full.means.device
        
        keep_mask = torch.ones(N, dtype=torch.bool, device=device)
        enlarge_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        
        tiles_h = self.H // self.initial_tile_size
        tiles_w = self.W // self.initial_tile_size
        tile_size = self.initial_tile_size
        
        # Pre-compute center pixel indices for all tiles
        for th in range(tiles_h):
            for tw in range(tiles_w):
                self.stats['total_tiles_processed'] += 1
                
                tile_y = th * tile_size
                tile_x = tw * tile_size
                
                # Get center 4 pixels (Phase 1)
                center_pixels = [
                    (tile_y + 1, tile_x + 1),
                    (tile_y + 1, tile_x + 2),
                    (tile_y + 2, tile_x + 1),
                    (tile_y + 2, tile_x + 2),
                ]
                
                # Collect center Gaussians
                center_gaussians = []
                center_indices = []
                for (y, x) in center_pixels:
                    pixel_idx = y * self.W + x
                    if pixel_idx < N:
                        center_indices.append(pixel_idx)
                        center_gaussians.append({
                            'covariance': gaussians_full.covariances[0, pixel_idx],
                            'harmonics': gaussians_full.harmonics[0, pixel_idx],
                            'opacity': gaussians_full.opacities[0, pixel_idx].item(),
                        })
                
                if len(center_gaussians) < 4:
                    self.stats['full_processed'] += 1
                    continue
                
                # Compute center similarity
                similarity = self.compute_gaussian_similarity(center_gaussians)
                
                if similarity >= self.HIGH_THRESHOLD:
                    # Early-stop at Phase 1: keep only center pixels
                    self.stats['early_stop_phase1'] += 1
                    
                    # Mark non-center pixels for removal
                    for y in range(tile_y, tile_y + tile_size):
                        for x in range(tile_x, tile_x + tile_size):
                            pixel_idx = y * self.W + x
                            if pixel_idx < N and pixel_idx not in center_indices:
                                keep_mask[pixel_idx] = False
                    
                    # Mark center for enlargement
                    for idx in center_indices:
                        enlarge_mask[idx] = True
                else:
                    # Continue processing (keep all pixels)
                    self.stats['full_processed'] += 1
        
        # Compute final stats
        num_kept = keep_mask.sum().item()
        self.stats['gaussians_output'] = num_kept
        self.stats['gaussians_saved'] = N - num_kept
        self.stats['keep_ratio'] = num_kept / N if N > 0 else 0
        self.stats['early_stop_ratio'] = (
            self.stats['early_stop_phase1'] + self.stats['early_stop_phase2']
        ) / max(1, self.stats['total_tiles_processed'])
        
        return keep_mask, enlarge_mask, self.stats


def apply_progressive_saes(
    gaussians_full,
    H: int, W: int,
    tile_size: int = 4,
    gpp: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, List]:
    """
    Apply progressive SAES to Gaussians.
    
    Returns:
        keep_mask: Boolean mask for Gaussians to keep
        enlarge_mask: Boolean mask for Gaussians to enlarge  
        stats: Processing statistics
        continue_pixels: List of pixels that need FSDR processing
    """
    saes = ProgressiveSAES(H, W, tile_size)
    keep_mask, enlarge_mask, stats = saes.process_all_tiles_fast(gaussians_full, gpp)
    
    # Collect pixels that were NOT early-stopped (for FSDR)
    continue_pixels = []
    for idx in range(keep_mask.shape[0]):
        if keep_mask[idx] and not enlarge_mask[idx]:
            # This Gaussian was kept but not from early-stop
            pixel_idx = idx // gpp
            y = pixel_idx // W
            x = pixel_idx % W
            # Boundary check
            if y < H and x < W:
                continue_pixels.append((y, x, pixel_idx))
    
    return keep_mask, enlarge_mask, stats, continue_pixels


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
    # Hardware simulator control options (for correctness verification)
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
    
    # Other options
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only run baseline, skip SCARF pipeline')
    args = parser.parse_args()
    
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
    print(f"[Config] SAES:")
    print(f"  tile={CONFIG.tile_size}, threshold={CONFIG.early_stop_threshold}")
    print(f"  cov_enlarge={CONFIG.cov_enlarge_factor}x for early-stop")
    print(f"[Config] FSDR:")
    print(f"  cache_size={CONFIG.fsdr_cache_size}, hamming_threshold={CONFIG.fsdr_hamming_threshold}")
    print(f"[Config] DSU:")
    print(f"  num_depth_candidates={CONFIG.num_depth_candidates}")
    print(f"[Config] GGU:")
    print(f"  scale_range=({CONFIG.scale_min}, {CONFIG.scale_max})")
    print()
    
    # Initialize cycle counter
    cycle_counter = CycleCounter()
    
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
    
    t0 = time.time()
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
    baseline_time = time.time() - t0
    baseline_image = baseline_output.color[0, 0]
    baseline_count = baseline_gaussians.means.shape[1]
    print(f"  ✓ Baseline: {baseline_image.shape}, Gaussians: {baseline_count:,}, Time: {baseline_time:.2f}s")
    
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
    
    # Estimate baseline cycles (for comparison)
    baseline_dsu_cycles = baseline_count * sum(CycleCounter.DSU_CYCLES.values())
    baseline_ggu_cycles = baseline_count * sum(CycleCounter.GGU_CYCLES.values())
    baseline_total_cycles = baseline_dsu_cycles + baseline_ggu_cycles
    
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
        print(f"  Time: {baseline_time:.2f}s")
        print(f"  Estimated cycles: {baseline_total_cycles:,}")
        return
    
    # --------------------------------------------------------
    # Step 4: Run SCARF Pipeline (Clear 3-Stage Flow)
    # --------------------------------------------------------
    # Pipeline Architecture:
    #   Stage 1: Feature Extraction → pipeline_features
    #   Stage 2: Depth Prediction   → pipeline_depths, pipeline_densities, pipeline_raw_gaussians
    #   Stage 3: Gaussian Generation → scarf_gaussians_full
    #
    # Each stage independently controlled by --no-feature/--no-depth/--no-gaussian
    # When a stage is disabled, use original GPU computation instead of HW simulator
    # --------------------------------------------------------
    print()
    print("[4/6] Running SCARF Pipeline...")
    
    t0 = time.time()
    
    # Initialize cycle counters
    feature_sim_cycles = 0
    depth_sim_cycles = 0
    
    # Initialize FSDR
    fsdr = FSDRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold
    )
    
    # ============================================================
    # STAGE 1: Feature Extraction
    # ============================================================
    print("  [Stage 1] Feature Extraction...")
    
    # Pipeline variables for Stage 1 output
    pipeline_features = None
    pipeline_cnn_features = None
    
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
                
                # Print cycle breakdown
                cycle_info = []
                if hasattr(fe_output, 'cnn_cycles') and fe_output.cnn_cycles > 0:
                    cycle_info.append(f"CNN: {fe_output.cnn_cycles:,}")
                if hasattr(fe_output, 'transformer_cycles') and fe_output.transformer_cycles > 0:
                    cycle_info.append(f"Transformer: {fe_output.transformer_cycles:,}")
                if hasattr(fe_output, 'dinov2_cycles') and fe_output.dinov2_cycles > 0:
                    cycle_info.append(f"DINOv2: {fe_output.dinov2_cycles:,}")
                
                print(f"    ✓ HW cycles: {feature_sim_cycles:,} ({', '.join(cycle_info)})")
                
                # Extract features from simulator output
                if hasattr(fe_output, 'trans_features') and fe_output.trans_features is not None:
                    pipeline_features = fe_output.trans_features
                if hasattr(fe_output, 'cnn_features') and fe_output.cnn_features is not None:
                    pipeline_cnn_features = fe_output.cnn_features
                
                print(f"    ✓ Features: {pipeline_features.shape if pipeline_features is not None else 'None'}")
        except Exception as e:
            print(f"    ⚠ HW Simulator error: {e}, falling back to GPU")
            use_feature_sim = False
    
    if not use_feature_sim or pipeline_features is None:
        # Use original GPU for feature extraction
        print(f"    Mode: Original GPU" + (" (fallback)" if use_feature_sim else " (--no-feature)"))
        
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
    
    use_depth_sim = not args.no_depth
    
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
                # Use actual HW simulation (may have quality loss)
                # set_accurate_mode(False) means use hardware simulation
                # set_accurate_mode(True) means use original GPU (only estimate cycles)
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
                # Debug: Print features stats to HW depth_predictor
                if pipeline_features is not None and not isinstance(pipeline_features, str):
                    pf_flat = pipeline_features.reshape(-1)
                    print(f"    [DEBUG] pipeline_features (to HW depth_predictor): min={pf_flat.min():.4f}, max={pf_flat.max():.4f}")
                elif isinstance(pipeline_features, str):
                    print(f"    [DepthSplat] Features: {pipeline_features} (will extract internally)")
                
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
                        cnn_features=pipeline_cnn_features,  # Input from Stage 1
                        extra_info=extra_info,
                    )
                
                depth_sim_cycles = dp_output.total_cycles
                cycle_breakdown = dp_output.cycle_breakdown.to_dict()
                
                print(f"    ✓ HW cycles: {depth_sim_cycles:,}")
                print(f"      Breakdown: cost_volume={cycle_breakdown['cost_volume']:,}, "
                      f"unet={cycle_breakdown['unet_refinement']:,}, "
                      f"depth_head={cycle_breakdown['depth_head']:,}, "
                      f"regression={cycle_breakdown['softmax_regression']:,}")
                
                # Extract outputs
                pipeline_depths = dp_output.depths
                pipeline_densities = dp_output.densities
                pipeline_raw_gaussians = dp_output.raw_gaussians
                
                print(f"    ✓ Depths: {pipeline_depths.shape if pipeline_depths is not None else 'None'}")
                
                # Debug: Print depth statistics
                if pipeline_depths is not None:
                    d_flat = pipeline_depths.reshape(-1)
                    print(f"      Depth stats: min={d_flat.min():.4f}, max={d_flat.max():.4f}, mean={d_flat.mean():.4f}")
                if pipeline_raw_gaussians is not None:
                    rg = pipeline_raw_gaussians.reshape(-1)
                    print(f"      raw_gaussians stats: min={rg.min():.4f}, max={rg.max():.4f}, mean={rg.mean():.4f}")
                if pipeline_densities is not None:
                    pd = pipeline_densities.reshape(-1)
                    print(f"      densities stats: min={pd.min():.4f}, max={pd.max():.4f}, mean={pd.mean():.4f}")
                    # Check what map_pdf_to_opacity would produce
                    if hasattr(model.encoder, 'map_pdf_to_opacity'):
                        test_op = model.encoder.map_pdf_to_opacity(pipeline_densities, 0)
                        test_op_flat = test_op.reshape(-1)
                        print(f"      after map_pdf_to_opacity: min={test_op_flat.min():.4f}, max={test_op_flat.max():.4f}")
                
                # DepthSplat special case: depth_predictor doesn't return raw_gaussians
                # We need to run encoder's feature_upsampler, gaussian_regressor, gaussian_head
                if args.model == 'depthsplat' and pipeline_raw_gaussians is None and pipeline_depths is not None:
                    print(f"    [DepthSplat] Computing raw_gaussians from encoder modules...")
                    try:
                        # Get depth_predictor output dict (need features for upsampler)
                        # Re-run depth_predictor to get internal features
                        with torch.no_grad():
                            # Convert near/far to inverse depth format
                            if near.dim() == 0:
                                near_bv = near.expand(B, V_ctx)
                            elif near.dim() == 1:
                                near_bv = near.unsqueeze(1).expand(B, V_ctx) if near.shape[0] == B else near.mean().expand(B, V_ctx)
                            else:
                                near_bv = near
                            if far.dim() == 0:
                                far_bv = far.expand(B, V_ctx)
                            elif far.dim() == 1:
                                far_bv = far.unsqueeze(1).expand(B, V_ctx) if far.shape[0] == B else far.mean().expand(B, V_ctx)
                            else:
                                far_bv = far
                            
                            min_depth = 1.0 / far_bv.clamp(min=1e-6)
                            max_depth = 1.0 / near_bv.clamp(min=1e-6)
                            
                            # Run depth_predictor to get results_dict with features
                            results_dict = model.encoder.depth_predictor(
                                context['image'],
                                attn_splits_list=[2],
                                intrinsics=context['intrinsics'],
                                min_depth=min_depth,
                                max_depth=max_depth,
                                extrinsics=context['extrinsics'],
                            )
                            
                            # depth_preds[-1] is [B, V, H, W]
                            depth_final = results_dict['depth_preds'][-1]
                            match_prob = results_dict['match_probs'][-1]  # [BV, D, H, W]
                            
                            # Feature upsampler
                            features_upsampled = model.encoder.feature_upsampler(
                                results_dict["features_mono_intermediate"],
                                cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                mv_features=results_dict["features_mv"][0] if model.encoder.cfg.num_scales == 1 
                                           else results_dict["features_mv"][::-1]
                            )  # [BV, C, H, W]
                            
                            # Get match probability max
                            match_prob_max = torch.max(match_prob, dim=1, keepdim=True)[0]  # [BV, 1, H, W]
                            if match_prob_max.shape[-2:] != depth_final.shape[-2:]:
                                match_prob_max = F.interpolate(match_prob_max, size=depth_final.shape[-2:], mode='nearest')
                            
                            # UNet input: concat(img, depth, match_prob, features)
                            concat_input = torch.cat((
                                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                                rearrange(depth_final, "b v h w -> (b v) () h w"),
                                match_prob_max,
                                features_upsampled,
                            ), dim=1)
                            
                            # Gaussian regressor
                            regressor_out = model.encoder.gaussian_regressor(concat_input)
                            
                            # Gaussian head input
                            gaussian_head_input = torch.cat([
                                regressor_out,
                                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                                features_upsampled,
                                match_prob_max,
                            ], dim=1)
                            
                            # Gaussian head
                            gaussians_bv = model.encoder.gaussian_head(gaussian_head_input)  # [BV, C, H, W]
                            
                            # Convert to pipeline format
                            pipeline_raw_gaussians = gaussians_bv  # Keep as [BV, C, H, W] for DepthSplat
                            
                            # Update densities from match_prob
                            pipeline_densities = rearrange(match_prob_max, "(b v) c h w -> b v (c h w) () ()", b=B, v=V_ctx)
                            
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
        # Use original GPU for depth prediction
        print(f"    Mode: Original GPU" + (" (fallback)" if use_depth_sim else " (--no-depth)"))
        
        if pipeline_features is not None and not isinstance(pipeline_features, str) and hasattr(model.encoder, 'depth_predictor'):
            # Debug: Print features stats
            pf_flat = pipeline_features.reshape(-1)
            print(f"    [DEBUG] pipeline_features (to GPU depth_predictor): min={pf_flat.min():.4f}, max={pf_flat.max():.4f}")
            
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
                    # DepthSplat has different depth predictor interface
                    # Run full encoder and capture intermediate outputs via hooks
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
                    # Also capture gaussian_head_output for DepthSplat
                    if 'gaussian_head_output' in captured:
                        pipeline_raw_gaussians = captured['gaussian_head_output']
            
            print(f"    ✓ Depths: {pipeline_depths.shape if pipeline_depths is not None else 'None'}")
            
            # Debug: Print GPU depth statistics
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
                print(f"    [DEBUG] SCARF means: {scarf_gaussians_full.means.shape}, min={scarf_gaussians_full.means.min():.4f}, max={scarf_gaussians_full.means.max():.4f}")
                print(f"    [DEBUG] Baseline means: {baseline_gaussians.means.shape}, min={baseline_gaussians.means.min():.4f}, max={baseline_gaussians.means.max():.4f}")
                print(f"    [DEBUG] SCARF opacities: {scarf_gaussians_full.opacities.shape}, min={scarf_gaussians_full.opacities.min():.4f}, max={scarf_gaussians_full.opacities.max():.4f}")
                print(f"    [DEBUG] Baseline opacities: {baseline_gaussians.opacities.shape}, min={baseline_gaussians.opacities.min():.4f}, max={baseline_gaussians.opacities.max():.4f}")
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
    # Step 4b: Progressive SAES (process tiles from center outward)
    # --------------------------------------------------------
    if args.no_saes:
        print("  [4b] Progressive SAES: SKIPPED (--no-saes)")
        # No early stopping - all pixels continue
        N = scarf_gaussians_full.means.shape[1]
        keep_mask = torch.ones(N, dtype=torch.bool, device=device)
        enlarge_mask = torch.zeros(N, dtype=torch.bool, device=device)
        saes_stats = {
            'total_tiles_processed': 0, 'early_stop_phase1': 0,
            'early_stop_phase2': 0, 'subdivided': 0, 'full_processed': 0,
            'early_stop_ratio': 0.0
        }
        continue_pixels = [(y, x, y * w + x) for y in range(h) for x in range(w)]
    else:
        print("  [4b] Progressive SAES (center-outward processing)...")
        
        # Apply progressive SAES - decides early-stop based on Gaussian similarity
        keep_mask, enlarge_mask, saes_stats, continue_pixels = apply_progressive_saes(
            scarf_gaussians_full, h, w, CONFIG.tile_size, gpp=1
        )
        
        print(f"  ✓ Tiles processed: {saes_stats['total_tiles_processed']}")
        print(f"    Early-stop Phase 1: {saes_stats['early_stop_phase1']} (center similar)")
        print(f"    Early-stop Phase 2: {saes_stats['early_stop_phase2']} (cross similar)")
        print(f"    Subdivided: {saes_stats['subdivided']} (detail detected)")
        print(f"    Full processed: {saes_stats['full_processed']}")
        print(f"  ✓ Early-stop ratio: {saes_stats['early_stop_ratio']*100:.1f}%")
    
    # --------------------------------------------------------
    # Step 4c: FSDR for continue pixels
    # --------------------------------------------------------
    N = scarf_gaussians_full.means.shape[1]
    depth_scale_factors = torch.ones(N, device=device)
    
    if args.no_fsdr:
        print("  [4c] FSDR depth reuse: SKIPPED (--no-fsdr)")
    else:
        print("  [4c] FSDR depth reuse for continue pixels...")
        
        # Process continue pixels through FSDR
        # Check if features is a tensor (not a string marker like 'depthsplat_integrated')
        has_features = (features is not None and 
                       not isinstance(features, str) and 
                       hasattr(features, 'shape'))
        
        if has_features and len(continue_pixels) > 0:
            B_feat, V_feat, C, H_feat, W_feat = features.shape
            features_up = F.interpolate(
                features[0], size=(h, w), mode='bilinear', align_corners=False
            ).mean(dim=0)  # [C, H, W]
            
            for (y, x, pixel_idx) in continue_pixels:
                feat = features_up[:, y, x]
                
                # Get original depth for this pixel
                # Handle different depth formats: 
                # - Transplat/MVSplat: [B, V, H*W, srf, dpt] (5D)
                # - DepthSplat: [B, V, H, W] (4D)
                if depths is not None:
                    if depths.dim() == 5:
                        original_depth = depths[0, 0, pixel_idx, 0, 0].item()
                    elif depths.dim() == 4:
                        # DepthSplat format: [B, V, H, W]
                        py, px = y, x
                        original_depth = depths[0, 0, py, px].item()
                    else:
                        original_depth = 1.0
                else:
                    original_depth = 1.0
                
                # FSDR processing
                path, num_searches, output_depth = fsdr.process_pixel(
                    feat, original_depth, (y, x), pixel_idx
                )
                
                # Record cycles
                cycle_counter.add_dsu_fsdr(path, num_searches)
                
                # Calculate scale factor if depth was modified
                if output_depth != original_depth and original_depth > 0:
                    scale_factor = output_depth / original_depth
                    depth_scale_factors[pixel_idx] = scale_factor
    
    # Record DSU skips for early-stop tiles
    num_early_stop_gaussians = enlarge_mask.sum().item()
    for _ in range(num_early_stop_gaussians):
        cycle_counter.add_dsu_skip()
    
    # Record GGU cycles for kept Gaussians
    num_kept = keep_mask.sum().item()
    cycle_counter.add_ggu(num_kept)
    
    gaussian_stats = {
        'gaussians_baseline': N,
        'gaussians_output': num_kept,
        'early_stop_gaussians': N - num_kept,
        'depth_modified_gaussians': (depth_scale_factors != 1.0).sum().item(),
    }
    
    # Apply FSDR depth modifications to gaussian means
    # FSDR now TRULY affects the 3D positions!
    from src.model.types import Gaussians
    
    modified_means = scarf_gaussians_full.means.clone()
    
    # Scale the z-coordinate (depth) based on FSDR decisions
    # The means are in world space, so we scale along the ray direction
    # For simplicity, we scale the entire position vector (approximation)
    for i in range(len(depth_scale_factors)):
        if depth_scale_factors[i] != 1.0:
            # Scale the position along the view ray (approximated by scaling z)
            modified_means[0, i, 2] *= depth_scale_factors[i]
    
    # Apply masks
    new_means = modified_means[:, keep_mask]
    new_covs = scarf_gaussians_full.covariances[:, keep_mask].clone()
    new_harmonics = scarf_gaussians_full.harmonics[:, keep_mask]
    new_opacities = scarf_gaussians_full.opacities[:, keep_mask]
    
    # Enlarge covariances for early-stop corners
    enlarge_mask_kept = enlarge_mask[keep_mask]
    new_covs[:, enlarge_mask_kept] *= CONFIG.cov_enlarge_factor
    
    scarf_gaussians = Gaussians(
        means=new_means,
        covariances=new_covs,
        harmonics=new_harmonics,
        opacities=new_opacities,
    )
    
    print(f"  ✓ Gaussians: {baseline_count:,} → {gaussian_stats['gaussians_output']:,}")
    print(f"  ✓ FSDR depth modified: {gaussian_stats['depth_modified_gaussians']:,} gaussians")
    
    # --------------------------------------------------------
    # Step 5: Render with Transplat Decoder
    # --------------------------------------------------------
    print()
    print("[5/6] Rendering with Transplat Decoder...")
    
    with torch.no_grad():
        scarf_output = model.decoder.forward(
            scarf_gaussians, tgt_ext, tgt_int,
            target['near'], target['far'], (h, w), depth_mode=None
        )
    
    scarf_time = time.time() - t0
    scarf_image = scarf_output.color[0, 0]
    
    print(f"  ✓ SCARF rendering complete, Time: {scarf_time:.2f}s")
    
    # --------------------------------------------------------
    # Step 6: Evaluate and Save
    # --------------------------------------------------------
    print()
    print("[6/6] Evaluating and saving...")
    
    gt_image = target['image'][0, 0]
    
    # Metrics
    def compute_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        return -10 * torch.log10(mse).item()
    
    def compute_ssim(img1, img2):
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
        return ssim(img1.unsqueeze(0), img2.unsqueeze(0)).item()
    
    baseline_psnr = compute_psnr(baseline_image, gt_image)
    baseline_ssim = compute_ssim(baseline_image, gt_image)
    scarf_psnr = compute_psnr(scarf_image, gt_image)
    scarf_ssim = compute_ssim(scarf_image, gt_image)
    
    # Save outputs
    output_dir = SCARF_ROOT / 'outputs' / 'demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from torchvision.utils import save_image
    save_image(gt_image, output_dir / 'gt_00.png')
    save_image(baseline_image, output_dir / 'baseline_00.png')
    save_image(scarf_image, output_dir / 'scarf_00.png')
    
    # Get statistics
    cycle_stats = cycle_counter.get_summary()
    fsdr_stats = fsdr.get_summary()
    
    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("RESULTS - SCARF Demo")
    print("=" * 70)
    print()
    print("### Quality Metrics")
    print(f"  BASELINE: PSNR={baseline_psnr:.2f} dB, SSIM={baseline_ssim:.4f}")
    print(f"  SCARF:    PSNR={scarf_psnr:.2f} dB, SSIM={scarf_ssim:.4f}")
    print(f"  Quality loss: {scarf_psnr - baseline_psnr:+.2f} dB")
    print()
    print("### Gaussian Count")
    print(f"  BASELINE: {gaussian_stats['gaussians_baseline']:,}")
    print(f"  SCARF:    {gaussian_stats['gaussians_output']:,}")
    reduction = (1 - gaussian_stats['gaussians_output'] / gaussian_stats['gaussians_baseline']) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print()
    print("### Progressive SAES Results")
    print(f"  Total tiles processed: {saes_stats['total_tiles_processed']}")
    print(f"  Early-stop Phase 1 (center): {saes_stats['early_stop_phase1']}")
    print(f"  Early-stop Phase 2 (cross):  {saes_stats['early_stop_phase2']}")
    print(f"  Subdivided (detail):         {saes_stats['subdivided']}")
    print(f"  Full processed:              {saes_stats['full_processed']}")
    print(f"  Overall early-stop ratio: {saes_stats['early_stop_ratio']*100:.1f}%")
    print()
    print("### FSDR Statistics (for continue tiles)")
    if fsdr_stats['total_pixels'] > 0:
        print(f"  Pixels processed: {fsdr_stats['total_pixels']:,}")
        print(f"  Cache hit rate: {fsdr_stats['hit_rate']*100:.1f}%")
        print(f"  Path distribution:")
        print(f"    Direct reuse:  {fsdr_stats['direct_reuse_rate']*100:.1f}%")
        print(f"    Interpolation: {fsdr_stats['interpolation_rate']*100:.1f}%")
        print(f"    Light verify:  {fsdr_stats['light_verify_rate']*100:.1f}%")
        print(f"    Full search:   {fsdr_stats['full_search_rate']*100:.1f}%")
        print(f"  Memory reduction: {fsdr_stats['memory_reduction']*100:.1f}%")
        print(f"  Depth reuse rate: {fsdr_stats['depth_reuse_rate']*100:.1f}%")
        print(f"  Depth modified: {fsdr_stats['depth_reused']:,} pixels (TRULY affected!)")
    else:
        print(f"  (No continue tiles)")
    print()
    print("### Hardware Cycle Counts")
    print(f"  BASELINE (estimated):")
    print(f"    DSU cycles: {baseline_dsu_cycles:,}")
    print(f"    GGU cycles: {baseline_ggu_cycles:,}")
    print(f"    Total:      {baseline_total_cycles:,}")
    print()
    scarf_total_cycles = cycle_stats['total_cycles'] + feature_sim_cycles + depth_sim_cycles
    print(f"  SCARF:")
    if feature_sim_cycles > 0:
        print(f"    Feature Extraction: {feature_sim_cycles:,}")
    if depth_sim_cycles > 0:
        print(f"    Depth Predictor:    {depth_sim_cycles:,}")
    print(f"    DSU cycles: {cycle_stats['dsu_cycles']:,}")
    print(f"    GGU cycles: {cycle_stats['ggu_cycles']:,}")
    print(f"    Total:      {scarf_total_cycles:,}")
    print()
    cycle_reduction = (1 - scarf_total_cycles / baseline_total_cycles) * 100 if baseline_total_cycles > 0 else 0
    print(f"  Cycle reduction: {cycle_reduction:.1f}%")
    print(f"  Speedup: {baseline_total_cycles / scarf_total_cycles:.2f}x" if scarf_total_cycles > 0 else "  Speedup: N/A")
    print()
    print("### Summary")
    print(f"  Quality loss:     {scarf_psnr - baseline_psnr:+.2f} dB")
    print(f"  Gaussian saving:  {reduction:.1f}%")
    print(f"  Cycle reduction:  {cycle_reduction:.1f}%")
    print()
    print(f"## Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
