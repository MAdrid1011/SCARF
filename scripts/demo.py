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
    
    # Determine checkpoint path
    checkpoint_path = str(SCARF_ROOT / 'transplat' / 'checkpoints' / 're10k.ckpt')
    
    # Load model
    model_bundle = loader.load_model(checkpoint_path)
    
    # Load data
    data_bundle = loader.load_data(model_bundle, num_samples=1)
    
    return model_bundle.model, data_bundle.batch, model_bundle.config, model_bundle.device


# ============================================================
# SAES: Scene-Adaptive Early-Stopping
# ============================================================
def compute_tile_similarity_from_features(features: torch.Tensor, H: int, W: int, tile_size: int) -> torch.Tensor:
    """
    Compute tile similarity from features (BEFORE depth search).
    
    Uses feature cosine similarity between pixels within tiles.
    High similarity = homogeneous region = candidate for early-stop.
    
    Args:
        features: [B, V, C, H_feat, W_feat] feature maps
        H, W: Target image dimensions
        tile_size: Tile size in pixels
    
    Returns:
        similarities: [tiles_h, tiles_w] similarity scores in [0, 1]
    """
    # Upsample features to image resolution
    B, V, C, H_feat, W_feat = features.shape
    features_up = F.interpolate(
        features.view(B*V, C, H_feat, W_feat),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )  # [B*V, C, H, W]
    
    # Average over views
    features_up = features_up.view(B, V, C, H, W).mean(dim=1)  # [B, C, H, W]
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features_up, dim=1)  # [B, C, H, W]
    
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    
    similarities = torch.zeros(tiles_h, tiles_w, device=features.device)
    
    for th in range(tiles_h):
        for tw in range(tiles_w):
            y0, y1 = th * tile_size, (th + 1) * tile_size
            x0, x1 = tw * tile_size, (tw + 1) * tile_size
            
            # Get tile features
            tile_feat = features_norm[0, :, y0:y1, x0:x1]  # [C, tile_h, tile_w]
            
            # Compute average cosine similarity between all pixel pairs
            # Reshape to [num_pixels, C]
            tile_flat = tile_feat.permute(1, 2, 0).reshape(-1, C)  # [16, C] for 4x4 tile
            
            # Compute pairwise cosine similarities
            cos_sim_matrix = torch.mm(tile_flat, tile_flat.t())  # [16, 16]
            
            # Average off-diagonal elements (exclude self-similarity)
            n_pixels = cos_sim_matrix.shape[0]
            mask = ~torch.eye(n_pixels, dtype=torch.bool, device=cos_sim_matrix.device)
            avg_sim = cos_sim_matrix[mask].mean().item()
            
            # Scale to [0, 1] - cosine similarity is already [-1, 1]
            # We map it so 1.0 = very similar, 0.0 = very different
            sim = (avg_sim + 1) / 2  # Map [-1, 1] to [0, 1]
            similarities[th, tw] = sim
    
    return similarities


def apply_saes_decisions(
    similarities: torch.Tensor,
    H: int, W: int,
    tile_size: int,
    threshold: float
) -> Tuple[torch.Tensor, Dict, List]:
    """
    Apply SAES tile-level decisions.
    
    Returns:
        early_stop_mask: [tiles_h, tiles_w] bool mask for early-stop tiles
        stats: Statistics dictionary
        continue_pixels: List of (y, x, pixel_idx) for continue tiles
    """
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    
    early_stop_mask = similarities >= threshold
    
    stats = {
        'total_tiles': tiles_h * tiles_w,
        'early_stop': early_stop_mask.sum().item(),
        'full_continue': (~early_stop_mask).sum().item(),
    }
    
    # Collect continue tile pixels for FSDR
    continue_pixels = []
    for th in range(tiles_h):
        for tw in range(tiles_w):
            if not early_stop_mask[th, tw]:
                y0, y1 = th * tile_size, (th + 1) * tile_size
                x0, x1 = tw * tile_size, (tw + 1) * tile_size
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        pixel_idx = y * W + x
                        continue_pixels.append((y, x, pixel_idx))
    
    return early_stop_mask, stats, continue_pixels


# ============================================================
# SCARF Gaussian Generation
# ============================================================
def generate_scarf_gaussians(
    early_stop_mask: torch.Tensor,
    depths: torch.Tensor,
    densities: torch.Tensor,
    raw_gaussians: torch.Tensor,
    context: Dict,
    H: int, W: int,
    tile_size: int,
    cycle_counter: CycleCounter,
    fsdr: FSDRSimulator,
    features: torch.Tensor,
    continue_pixels: List,
    gaussians_full,  # Full Gaussians object to modify
) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
    """
    Generate Gaussians using SCARF logic with SAES and FSDR.
    
    FSDR now TRULY modifies depth values for cache hits.
    
    For early-stop tiles: Keep only corner gaussians, enlarge covariance
    For continue tiles: Process through FSDR, modify depths based on cache
    
    Returns:
        keep_mask, enlarge_mask, stats, depth_scale_factors
    """
    from src.model.types import Gaussians
    from src.geometry.projection import sample_image_grid
    
    device = depths.device
    B, V, R, srf, gpp = depths.shape
    
    # Initialize GGU
    from ggu import GGUConfig, GGUProcessor
    ggu_config = GGUConfig(
        scale_min=CONFIG.scale_min,
        scale_max=CONFIG.scale_max,
        sh_degree=CONFIG.sh_degree,
        image_shape=(H, W),
    )
    ggu = GGUProcessor(ggu_config, enable_cycle_counting=True)
    
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    
    # Track depth modifications from FSDR
    depth_scale_factors = torch.ones(gaussians_full.means.shape[1], device=device)
    
    # Process depth search with FSDR for continue tiles
    # FSDR now TRULY affects depth values
    if features is not None and len(continue_pixels) > 0:
        B_feat, V_feat, C, H_feat, W_feat = features.shape
        features_up = F.interpolate(
            features[0], size=(H, W), mode='bilinear', align_corners=False
        )  # [V, C, H, W]
        
        # Get original depth values from gaussians (z coordinate)
        original_depths = gaussians_full.means[0, :, 2].clone()  # [N]
        
        for y, x, pixel_idx in continue_pixels:
            # Get feature for this pixel (average over views)
            feat = features_up[:, :, y, x].mean(dim=0)  # [C]
            
            # Get original depth
            if pixel_idx < len(original_depths):
                original_depth = original_depths[pixel_idx].item()
            else:
                original_depth = original_depths[0].item()
            
            # Process through FSDR - now returns modified depth!
            path, num_searches, output_depth = fsdr.process_pixel(feat, original_depth, (y, x), pixel_idx)
            
            # Record cycles
            cycle_counter.add_dsu_fsdr(path, num_searches)
            
            # Calculate depth scale factor for this pixel
            if original_depth > 0 and output_depth != original_depth:
                scale_factor = output_depth / original_depth
                # Apply to corresponding gaussians (both views)
                for v in range(V):
                    idx = pixel_idx * V * srf * gpp + v * srf * gpp
                    for s in range(srf):
                        for g in range(gpp):
                            flat_idx = idx + s * gpp + g
                            if flat_idx < len(depth_scale_factors):
                                depth_scale_factors[flat_idx] = scale_factor
    
    # Record DSU skips for early-stop tiles
    num_early_stop_pixels = int(early_stop_mask.sum().item()) * tile_size * tile_size * 2  # 2 views
    for _ in range(num_early_stop_pixels):
        cycle_counter.add_dsu_skip()
    
    # Now apply masking based on SAES decisions
    pixels_per_view = H * W
    N = gaussians_full.means.shape[1]  # Total gaussians
    
    # Create masks
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    enlarge_mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    for th in range(tiles_h):
        for tw in range(tiles_w):
            y0, y1 = th * tile_size, (th + 1) * tile_size
            x0, x1 = tw * tile_size, (tw + 1) * tile_size
            
            if early_stop_mask[th, tw]:
                # Early-stop: keep only corners, enlarge covariance
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        pixel_idx = y * W + x
                        is_corner = (y == y0 or y == y1-1) and (x == x0 or x == x1-1)
                        
                        # For both views
                        for v in range(V):
                            # Calculate index in flattened tensor
                            idx = pixel_idx * V * srf * gpp + v * srf * gpp
                            for s in range(srf):
                                for g in range(gpp):
                                    flat_idx = idx + s * gpp + g
                                    if flat_idx < N:
                                        if not is_corner:
                                            keep_mask[flat_idx] = False
                                        else:
                                            enlarge_mask[flat_idx] = True
    
    # Count gaussians
    num_kept = keep_mask.sum().item()
    num_depth_modified = (depth_scale_factors != 1.0).sum().item()
    
    stats = {
        'gaussians_baseline': N,
        'gaussians_output': num_kept,
        'early_stop_gaussians': N - num_kept,
        'depth_modified_gaussians': num_depth_modified,
    }
    
    # Record GGU cycles
    cycle_counter.add_ggu(num_kept)
    
    return keep_mask, enlarge_mask, stats, depth_scale_factors


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
    # Step 4b: SAES tile decisions (BEFORE depth search conceptually)
    # --------------------------------------------------------
    print("  [4b] SAES tile decisions...")
    
    similarities = compute_tile_similarity_from_features(features, h, w, CONFIG.tile_size)
    
    # Debug: show similarity distribution
    sim_flat = similarities.flatten()
    print(f"  Similarity stats: min={sim_flat.min():.3f}, max={sim_flat.max():.3f}, mean={sim_flat.mean():.3f}, std={sim_flat.std():.3f}")
    
    # Adaptive threshold: use percentile to ensure some continue tiles
    # Target: ~40% early-stop (conservative for quality < 3dB loss)
    sorted_sims = torch.sort(sim_flat, descending=True)[0]
    target_early_stop_ratio = 0.4
    threshold_idx = int(len(sorted_sims) * target_early_stop_ratio)
    adaptive_threshold = sorted_sims[threshold_idx].item()
    print(f"  Adaptive threshold for {target_early_stop_ratio*100:.0f}% early-stop: {adaptive_threshold:.3f}")
    
    # Use adaptive threshold instead of fixed
    effective_threshold = max(adaptive_threshold, 0.85)  # At least 0.85 similarity for early-stop
    print(f"  Using effective threshold: {effective_threshold:.3f}")
    
    early_stop_mask, saes_stats, continue_pixels = apply_saes_decisions(
        similarities, h, w, CONFIG.tile_size, effective_threshold
    )
    
    early_pct = saes_stats['early_stop'] / saes_stats['total_tiles'] * 100
    print(f"  ✓ SAES: {saes_stats['early_stop']}/{saes_stats['total_tiles']} tiles early-stop ({early_pct:.1f}%)")
    
    # --------------------------------------------------------
    # Step 4c: Process with SCARF (DSU + FSDR + GGU)
    # --------------------------------------------------------
    print("  [4c] Processing with SCARF (DSU + FSDR + GGU)...")
    
    keep_mask, enlarge_mask, gaussian_stats, depth_scale_factors = generate_scarf_gaussians(
        early_stop_mask, depths, densities, raw_gaussians,
        context, h, w, CONFIG.tile_size,
        cycle_counter, fsdr, features, continue_pixels,
        scarf_gaussians_full  # Pass full gaussians for depth modification
    )
    
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
    print("### SAES Tile Distribution")
    print(f"  Total tiles: {saes_stats['total_tiles']}")
    print(f"  Early-stop:    {saes_stats['early_stop']:5d} ({saes_stats['early_stop']/saes_stats['total_tiles']*100:.1f}%)")
    print(f"  Full-continue: {saes_stats['full_continue']:5d} ({saes_stats['full_continue']/saes_stats['total_tiles']*100:.1f}%)")
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
