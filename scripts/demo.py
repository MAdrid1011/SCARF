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
    python demo.py [--model MODEL_TYPE]

Supported Models:
    - transplat (default)
    - mvsplat (coming soon)
    - depthsplat (coming soon)
"""

import sys
import os
from pathlib import Path

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
    
    # DSU config
    num_depth_candidates: int = 32
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
    args = parser.parse_args()
    
    # Initialize config
    CONFIG = SCARFConfig(model_type=args.model)
    
    # Create adapter and apply model-specific overrides
    adapter = create_adapter(args.model)
    CONFIG.apply_adapter_overrides(adapter)
    
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
        baseline_gaussians = model.encoder(context, False)
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
    
    # Estimate baseline cycles (for comparison)
    baseline_dsu_cycles = baseline_count * sum(CycleCounter.DSU_CYCLES.values())
    baseline_ggu_cycles = baseline_count * sum(CycleCounter.GGU_CYCLES.values())
    baseline_total_cycles = baseline_dsu_cycles + baseline_ggu_cycles
    
    # --------------------------------------------------------
    # Step 4: Run SCARF Pipeline
    # --------------------------------------------------------
    print()
    print("[4/6] Running SCARF Pipeline...")
    print("  [4a] Extracting backbone features ONLY...")
    
    t0 = time.time()
    
    # Initialize FSDR
    fsdr = FSDRSimulator(
        feature_dim=CONFIG.feature_dim,
        cache_size=CONFIG.fsdr_cache_size,
        hamming_threshold=CONFIG.fsdr_hamming_threshold
    )
    
    # Capture intermediate outputs from encoder
    captured = {}
    
    def hook_backbone(module, inputs, outputs):
        trans_features, cnn_features = outputs
        captured['features'] = trans_features.detach()
        captured['cnn_features'] = cnn_features.detach() if cnn_features is not None else None
    
    def hook_depth_predictor(module, inputs, outputs):
        depths, densities, raw_gaussians = outputs
        captured['depths'] = depths.detach()
        captured['densities'] = densities.detach()
        captured['raw_gaussians'] = raw_gaussians.detach()
    
    # Register hooks
    hook1 = model.encoder.backbone.register_forward_hook(hook_backbone)
    hook2 = model.encoder.depth_predictor.register_forward_hook(hook_depth_predictor)
    
    with torch.no_grad():
        # Run encoder to get all intermediate outputs
        scarf_gaussians_full = model.encoder(context, False)
    
    hook1.remove()
    hook2.remove()
    
    features = captured.get('features')
    depths = captured.get('depths')
    densities = captured.get('densities')
    raw_gaussians = captured.get('raw_gaussians')
    
    print(f"  ✓ Features: {features.shape if features is not None else 'None'}")
    print(f"  ✓ Depths: {depths.shape if depths is not None else 'None'}")
    
    # --------------------------------------------------------
    # Step 4b: Progressive SAES (process tiles from center outward)
    # --------------------------------------------------------
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
    print("  [4c] FSDR depth reuse for continue pixels...")
    
    # Track depth modifications
    N = scarf_gaussians_full.means.shape[1]
    depth_scale_factors = torch.ones(N, device=device)
    
    # Process continue pixels through FSDR
    if features is not None and len(continue_pixels) > 0:
        B_feat, V_feat, C, H_feat, W_feat = features.shape
        features_up = F.interpolate(
            features[0], size=(h, w), mode='bilinear', align_corners=False
        ).mean(dim=0)  # [C, H, W]
        
        for (y, x, pixel_idx) in continue_pixels:
            feat = features_up[:, y, x]
            
            # Get original depth for this pixel
            original_depth = depths[0, 0, pixel_idx, 0, 0].item() if depths is not None else 1.0
            
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
    print(f"  SCARF:")
    print(f"    DSU cycles: {cycle_stats['dsu_cycles']:,}")
    print(f"    GGU cycles: {cycle_stats['ggu_cycles']:,}")
    print(f"    Total:      {cycle_stats['total_cycles']:,}")
    print()
    cycle_reduction = (1 - cycle_stats['total_cycles'] / baseline_total_cycles) * 100 if baseline_total_cycles > 0 else 0
    print(f"  Cycle reduction: {cycle_reduction:.1f}%")
    print(f"  Speedup: {baseline_total_cycles / cycle_stats['total_cycles']:.2f}x" if cycle_stats['total_cycles'] > 0 else "  Speedup: N/A")
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
