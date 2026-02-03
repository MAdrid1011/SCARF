"""
SAES Standalone Demo

Demonstrates how to use SAES TileProcessor with mock data.
Shows the 3-phase workflow and profiling output.
"""
import torch
import sys
from pathlib import Path

# Add SCARF to path
scarf_root = Path(__file__).parent.parent
sys.path.insert(0, str(scarf_root))

from saes import TileProcessor, TileConfig
from saes.types import Gaussian


def create_mock_depth_predictor():
    """Create mock depth predictor function"""
    def depth_fn(features, indices):
        # Mock: return random depths for requested indices
        # In real implementation, this would call transplat's depth_predictor
        return torch.rand(len(indices), 1) * 5.0 + 1.0  # Depths in [1, 6]
    return depth_fn


def create_mock_gaussian_adapter():
    """Create mock Gaussian adapter function"""
    def gaussian_fn(depths, context):
        # Mock: generate Gaussians from depths
        # In real implementation, this would call transplat's gaussian_adapter
        gaussians = []
        for depth in depths:
            mean = torch.randn(3) * depth.item()
            cov = torch.eye(3) * torch.rand(1).item() * 0.5
            opacity = 0.7 + torch.rand(1).item() * 0.2
            harmonics = torch.randn(3, 16)
            gaussians.append(Gaussian(mean, cov, opacity, harmonics))
        return gaussians
    return gaussian_fn


def main():
    """Run SAES standalone demo"""
    print("="*80)
    print("SAES Standalone Demo")
    print("="*80)
    
    # Configure SAES
    config = TileConfig(
        tile_size=4,
        probe_size=4,
        high_similarity_threshold=0.85,
        low_similarity_threshold=0.60,
    )
    
    print(f"\nConfiguration:")
    print(f"  Tile size: {config.tile_size}×{config.tile_size} = {config.tile_size**2} pixels")
    print(f"  Probe size: {config.probe_size} pixels")
    print(f"  High threshold: {config.high_similarity_threshold} (early-stop)")
    print(f"  Low threshold: {config.low_similarity_threshold} (full-continue)")
    
    # Create mock data
    B, V, C, H, W = 1, 2, 128, 16, 16  # Small feature map for demo
    features = torch.randn(B, V, C, H, W)
    context = {"scene_name": "demo_scene"}
    
    print(f"\nInput:")
    print(f"  Feature map: {B}×{V}×{C}×{H}×{W}")
    print(f"  Expected tiles: {H//config.tile_size}×{W//config.tile_size} = {(H//config.tile_size)**2} tiles")
    
    # Initialize processor
    processor = TileProcessor(config)
    
    # Create mock callbacks
    depth_fn = create_mock_depth_predictor()
    gaussian_fn = create_mock_gaussian_adapter()
    
    # Process scene
    print(f"\nProcessing scene...")
    gaussians, profiling = processor.process_scene(
        features, depth_fn, gaussian_fn, context
    )
    
    # Print results
    print(f"\n" + "="*80)
    print("Results:")
    print("="*80)
    print(profiling.summary_str())
    
    # Detailed path breakdown
    print(f"\nPath Distribution:")
    total = profiling.total_tiles
    print(f"  Early-stop:      {profiling.early_stop_tiles:3d} tiles ({profiling.early_stop_tiles/total:6.1%})")
    print(f"  Sparse-continue: {profiling.sparse_continue_tiles:3d} tiles ({profiling.sparse_continue_tiles/total:6.1%})")
    print(f"  Full-continue:   {profiling.full_continue_tiles:3d} tiles ({profiling.full_continue_tiles/total:6.1%})")
    
    # Savings breakdown
    total_pixels = total * config.tile_size ** 2
    print(f"\nComputation Savings:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Pixels saved: {profiling.total_pixels_saved}")
    print(f"  Saving ratio: {profiling.computation_saving_ratio:.2%}")
    
    # Timing
    print(f"\nTiming:")
    print(f"  SAES overhead: {profiling.overhead_ns / 1e6:.2f} ms")
    print(f"  Net speedup: {profiling.net_speedup:.2f}×")
    
    # Output
    print(f"\nOutput:")
    print(f"  Total Gaussians generated: {len(gaussians)}")
    
    print(f"\n" + "="*80)
    print("Demo complete!")
    print("="*80)


if __name__ == "__main__":
    main()
