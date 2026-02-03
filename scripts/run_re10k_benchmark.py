#!/usr/bin/env python3
"""
SCARF RE10K Benchmark CLI

Run SCARF inference benchmark on RE10K dataset.

Usage:
    python scripts/run_re10k_benchmark.py \
        --model-path checkpoints/re10k.ckpt \
        --dataset-path datasets/re10k/ \
        --output-dir outputs/benchmark/ \
        --num-scenes 100
"""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import BenchmarkRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SCARF benchmark on RE10K dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint (optional, uses simulation mode if not provided)',
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to RE10K dataset (optional, uses simulation mode if not provided)',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/benchmark/',
        help='Output directory for benchmark reports',
    )
    
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=100,
        help='Number of scenes to benchmark',
    )
    
    parser.add_argument(
        '--warmup-scenes',
        type=int,
        default=5,
        help='Number of warmup scenes (not included in metrics)',
    )
    
    parser.add_argument(
        '--per-scene',
        action='store_true',
        help='Collect per-scene metrics',
    )
    
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Force simulation mode (no real model/dataset)',
    )
    
    parser.add_argument(
        '--psnr-threshold',
        type=float,
        default=0.5,
        help='Maximum acceptable PSNR drop (dB)',
    )
    
    parser.add_argument(
        '--ssim-threshold',
        type=float,
        default=0.01,
        help='Maximum acceptable SSIM drop',
    )
    
    return parser.parse_args()


def progress_bar(current: int, total: int, prefix: str = ''):
    """Print progress bar."""
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '=' * filled + '-' * (bar_len - filled)
    print(f'\r{prefix}[{bar}] {current}/{total} ({pct:.1f}%)', end='', flush=True)
    if current == total:
        print()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("SCARF RE10K Benchmark")
    print("=" * 60)
    print()
    
    # Determine mode
    if args.simulation or args.model_path is None:
        print("Mode: SIMULATION (no real model/dataset)")
        model_path = None
        dataset_path = None
    else:
        print(f"Mode: REAL INFERENCE")
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.dataset_path}")
        model_path = args.model_path
        dataset_path = args.dataset_path
    
    print(f"Scenes: {args.num_scenes}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Initialize runner
    runner = BenchmarkRunner(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
    )
    
    # Run baseline
    print("Running baseline benchmark...")
    baseline = runner.run_baseline(
        num_scenes=args.num_scenes,
        collect_per_scene=args.per_scene,
        progress_callback=lambda c, t: progress_bar(c, t, 'Baseline: '),
    )
    print(f"  PSNR: {baseline.quality.psnr_mean:.2f} dB")
    print(f"  SSIM: {baseline.quality.ssim_mean:.4f}")
    print(f"  Time: {baseline.total_time_s:.2f}s")
    print()
    
    # Run SCARF
    print("Running SCARF benchmark...")
    scarf = runner.run_scarf(
        num_scenes=args.num_scenes,
        collect_per_scene=args.per_scene,
        progress_callback=lambda c, t: progress_bar(c, t, 'SCARF: '),
    )
    print(f"  PSNR: {scarf.quality.psnr_mean:.2f} dB")
    print(f"  SSIM: {scarf.quality.ssim_mean:.4f}")
    print(f"  Time: {scarf.total_time_s:.2f}s")
    print(f"  FSDR Hit Rate: {scarf.fsdr_stats['hit_rate']:.1%}")
    print(f"  Memory Reduction: {scarf.fsdr_stats['memory_reduction']:.1%}")
    print()
    
    # Compare
    print("Generating comparison report...")
    report = runner.compare_results(
        baseline=baseline,
        scarf=scarf,
        psnr_threshold=args.psnr_threshold,
        ssim_threshold=args.ssim_threshold,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Speedup: {report.speedup:.2f}×")
    print(f"PSNR Delta: {report.psnr_delta:+.3f} dB")
    print(f"SSIM Delta: {report.ssim_delta:+.4f}")
    print(f"Quality Acceptable: {'YES' if report.quality_acceptable else 'NO'}")
    print()
    
    # Save report
    runner.save_report(report, 'benchmark_report.json')
    print(f"Report saved to: {args.output_dir}/benchmark_report.json")
    print(f"Markdown saved to: {args.output_dir}/benchmark_report.md")
    
    # Exit code based on quality
    if report.quality_acceptable:
        print("\n✓ Benchmark passed!")
        return 0
    else:
        print("\n✗ Benchmark FAILED: Quality threshold exceeded")
        return 1


if __name__ == '__main__':
    sys.exit(main())
