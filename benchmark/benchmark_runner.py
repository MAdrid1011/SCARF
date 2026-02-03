"""
Benchmark Runner

End-to-end benchmark orchestration for SCARF.
"""
import os
import time
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from .cycle_counter import CycleCounter
from .quality_validator import QualityValidator, QualityMetrics
from .report_generator import ReportGenerator, BenchmarkReport, ComparisonReport

if TYPE_CHECKING:
    from .transplat_runner import TransplatRunner
    from .scarf_hooks import SCARFHooks


@dataclass
class BaselineResult:
    """Result from baseline (non-SCARF) benchmark run."""
    quality: QualityMetrics
    num_scenes: int
    total_time_s: float
    per_scene_metrics: Optional[Dict[str, QualityMetrics]] = None


@dataclass
class SCARFResult:
    """Result from SCARF-accelerated benchmark run."""
    quality: QualityMetrics
    cycles: CycleCounter
    fsdr_stats: Dict
    num_scenes: int
    total_time_s: float
    per_scene_metrics: Optional[Dict[str, QualityMetrics]] = None


class BenchmarkRunner:
    """
    End-to-end benchmark runner for SCARF.
    
    Orchestrates baseline and SCARF benchmark runs, collecting
    quality metrics and cycle counts for comparison.
    
    Example:
        runner = BenchmarkRunner(
            model_path='checkpoints/re10k.ckpt',
            dataset_path='datasets/re10k/',
            output_dir='outputs/benchmark/',
        )
        
        # Run benchmarks
        baseline = runner.run_baseline(num_scenes=100)
        scarf = runner.run_scarf(num_scenes=100)
        
        # Compare and save
        report = runner.compare_results(baseline, scarf)
        runner.save_report(report, 'benchmark_report.json')
    
    Note:
        This is a simulation-mode runner. For actual transplat integration,
        the model loading and inference would be connected to the real
        transplat codebase.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        scarf_config: Optional[Dict] = None,
        output_dir: str = 'outputs/benchmark/',
        use_real_inference: bool = False,
        device: str = 'cuda',
        save_images: bool = False,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            model_path: Path to model checkpoint
            dataset_path: Path to RE10K dataset
            scarf_config: SCARF configuration
            output_dir: Output directory for reports
            use_real_inference: Whether to use real model inference
            device: Device to use ('cuda' or 'cpu')
            save_images: Whether to save rendered images
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.scarf_config = scarf_config or {
            'enable_fsdr': True,
            'enable_saes': False,
        }
        self.output_dir = output_dir
        self.use_real_inference = use_real_inference
        self.device = device
        self.save_images = save_images
        
        # Report generator
        self.report_generator = ReportGenerator(model_name='transplat')
        
        # Simulation mode flag
        self._simulation_mode = not use_real_inference or model_path is None or not os.path.exists(model_path or '')
        
        # Real inference components (lazy-loaded)
        self._transplat_runner: Optional['TransplatRunner'] = None
        self._scarf_hooks: Optional['SCARFHooks'] = None
    
    def _get_transplat_runner(self) -> 'TransplatRunner':
        """Get or create TransplatRunner for real inference."""
        if self._transplat_runner is None:
            from .transplat_runner import TransplatRunner
            from .scarf_hooks import SCARFHooks
            
            # Create SCARF hooks if enabled
            if self.scarf_config.get('enable_fsdr', True):
                self._scarf_hooks = SCARFHooks(
                    enable_fsdr=True,
                    enable_cycle_counting=True,
                    cycle_counter=CycleCounter(),
                )
            
            self._transplat_runner = TransplatRunner(
                checkpoint_path=self.model_path,
                dataset_root=self.dataset_path,
                device=self.device,
                scarf_hooks=self._scarf_hooks,
            )
        
        return self._transplat_runner
    
    def run_baseline(
        self,
        num_scenes: int = 100,
        collect_per_scene: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BaselineResult:
        """
        Run baseline benchmark (without SCARF).
        
        Args:
            num_scenes: Number of scenes to process
            collect_per_scene: Whether to collect per-scene metrics
            progress_callback: Optional callback(current, total)
        
        Returns:
            BaselineResult with quality metrics
        """
        validator = QualityValidator()
        per_scene = {} if collect_per_scene else None
        
        start_time = time.time()
        
        for i in range(num_scenes):
            # In simulation mode, generate mock data
            if self._simulation_mode:
                rendered, gt = self._generate_mock_images()
            else:
                # Real mode: would load from dataset and run model
                rendered, gt = self._run_baseline_inference(i)
            
            validator.record_sample(rendered, gt)
            
            if collect_per_scene:
                scene_validator = QualityValidator()
                scene_validator.record_sample(rendered, gt)
                per_scene[f'scene_{i:04d}'] = scene_validator.get_metrics()
            
            if progress_callback:
                progress_callback(i + 1, num_scenes)
        
        total_time = time.time() - start_time
        
        return BaselineResult(
            quality=validator.get_metrics(),
            num_scenes=num_scenes,
            total_time_s=total_time,
            per_scene_metrics=per_scene,
        )
    
    def run_scarf(
        self,
        num_scenes: int = 100,
        collect_per_scene: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SCARFResult:
        """
        Run SCARF-accelerated benchmark.
        
        Args:
            num_scenes: Number of scenes to process
            collect_per_scene: Whether to collect per-scene metrics
            progress_callback: Optional callback(current, total)
        
        Returns:
            SCARFResult with quality metrics and cycle counts
        """
        validator = QualityValidator()
        cycle_counter = CycleCounter()
        per_scene = {} if collect_per_scene else None
        
        # FSDR statistics
        total_pixels = 0
        cache_hits = 0
        total_memory_accesses_baseline = 0
        total_memory_accesses_scarf = 0
        
        start_time = time.time()
        
        for i in range(num_scenes):
            # In simulation mode, generate mock data
            if self._simulation_mode:
                rendered, gt = self._generate_mock_images()
                scene_cycles, scene_fsdr = self._simulate_scarf_cycles()
            else:
                # Real mode: would run SCARF-accelerated inference
                rendered, gt, scene_cycles, scene_fsdr = self._run_scarf_inference(i)
            
            validator.record_sample(rendered, gt)
            
            # Aggregate FSDR stats
            total_pixels += scene_fsdr.get('total_pixels', 1000)
            cache_hits += scene_fsdr.get('cache_hits', 680)
            total_memory_accesses_baseline += scene_fsdr.get('baseline_accesses', 32000)
            total_memory_accesses_scarf += scene_fsdr.get('scarf_accesses', 10920)
            
            # Record cycles
            for component, ops in scene_cycles.items():
                for op, cycles in ops.items():
                    cycle_counter.record(component, op, cycles)
            
            if collect_per_scene:
                scene_validator = QualityValidator()
                scene_validator.record_sample(rendered, gt)
                per_scene[f'scene_{i:04d}'] = scene_validator.get_metrics()
            
            if progress_callback:
                progress_callback(i + 1, num_scenes)
        
        total_time = time.time() - start_time
        
        # Aggregate FSDR stats
        fsdr_stats = {
            'hit_rate': cache_hits / total_pixels if total_pixels > 0 else 0,
            'memory_reduction': 1 - (total_memory_accesses_scarf / total_memory_accesses_baseline)
                if total_memory_accesses_baseline > 0 else 0,
            'total_pixels': total_pixels,
            'cache_hits': cache_hits,
        }
        
        return SCARFResult(
            quality=validator.get_metrics(),
            cycles=cycle_counter,
            fsdr_stats=fsdr_stats,
            num_scenes=num_scenes,
            total_time_s=total_time,
            per_scene_metrics=per_scene,
        )
    
    def compare_results(
        self,
        baseline: BaselineResult,
        scarf: SCARFResult,
        psnr_threshold: float = 0.5,
        ssim_threshold: float = 0.01,
    ) -> ComparisonReport:
        """
        Compare baseline and SCARF results.
        
        Args:
            baseline: Baseline result
            scarf: SCARF result
            psnr_threshold: Maximum acceptable PSNR drop
            ssim_threshold: Maximum acceptable SSIM drop
        
        Returns:
            ComparisonReport with comparison metrics
        """
        baseline_report = self.report_generator.generate_baseline_report(
            quality=baseline.quality,
            num_scenes=baseline.num_scenes,
        )
        
        scarf_report = self.report_generator.generate_scarf_report(
            quality=scarf.quality,
            cycles=scarf.cycles,
            fsdr_stats=scarf.fsdr_stats,
            num_scenes=scarf.num_scenes,
        )
        
        return self.report_generator.generate_comparison(
            baseline=baseline_report,
            scarf=scarf_report,
            psnr_threshold=psnr_threshold,
            ssim_threshold=ssim_threshold,
        )
    
    def save_report(self, report: ComparisonReport, filename: str):
        """Save comparison report to file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        self.report_generator.save_json(report, filepath)
        
        # Also save markdown
        md_filepath = filepath.replace('.json', '.md')
        with open(md_filepath, 'w') as f:
            f.write(self.report_generator.to_markdown(report))
    
    # ============ Private Methods ============
    
    def _generate_mock_images(self) -> tuple:
        """Generate mock rendered and GT images for simulation."""
        torch.manual_seed(int(time.time() * 1000) % 2**32)
        gt = torch.rand(3, 256, 256)
        # Add small noise to simulate rendering differences
        noise = torch.randn_like(gt) * 0.02
        rendered = torch.clamp(gt + noise, 0, 1)
        return rendered, gt
    
    def _simulate_scarf_cycles(self) -> tuple:
        """Simulate SCARF cycle counts for a scene."""
        # Simulate ~1000 pixels per scene
        num_pixels = 1000
        hit_rate = 0.68
        
        cache_hits = int(num_pixels * hit_rate)
        cache_misses = num_pixels - cache_hits
        
        # Cycle breakdown
        cycles = {
            'fsdr': {
                'hash': num_pixels * 3,
                'lookup': num_pixels * 5,
                'direct_reuse': int(cache_hits * 0.5) * 1,
                'interpolation': int(cache_hits * 0.5) * 4,
            },
            'dsu': {
                'project': cache_misses * 32 * 8,
                'sample': cache_misses * 32 * 4,
                'cost': cache_misses * 32 * 2,
                'softmax': cache_misses * 6,
            },
            'ggu': {
                'position': num_pixels * 2,
                'covariance': num_pixels * 5,
                'sh_rotation': num_pixels * 3,
            },
        }
        
        # FSDR stats
        fsdr = {
            'total_pixels': num_pixels,
            'cache_hits': cache_hits,
            'baseline_accesses': num_pixels * 32,
            'scarf_accesses': cache_hits * 1 + cache_misses * 32,
        }
        
        return cycles, fsdr
    
    def _run_baseline_inference(self, scene_idx: int) -> tuple:
        """
        Run actual baseline inference using TransplatRunner.
        
        Args:
            scene_idx: Scene index
        
        Returns:
            Tuple of (rendered, ground_truth) tensors
        """
        runner = self._get_transplat_runner()
        
        # Create mock batch (placeholder - actual would come from dataset)
        batch = runner._create_mock_batch()
        
        # Run inference without SCARF
        result = runner.run_inference(batch, enable_scarf=False)
        
        # Save images if configured
        if self.save_images:
            runner._save_scene_images(result, self.output_dir)
        
        # Return single image pair (first target view)
        rendered = result.rendered[0]  # [C, H, W]
        gt = result.ground_truth[0]  # [C, H, W]
        
        return rendered, gt
    
    def _run_scarf_inference(self, scene_idx: int) -> tuple:
        """
        Run SCARF-accelerated inference using TransplatRunner.
        
        Args:
            scene_idx: Scene index
        
        Returns:
            Tuple of (rendered, ground_truth, cycles, fsdr_stats)
        """
        runner = self._get_transplat_runner()
        
        # Create mock batch (placeholder - actual would come from dataset)
        batch = runner._create_mock_batch()
        
        # Run inference with SCARF
        result = runner.run_inference(batch, enable_scarf=True)
        
        # Save images if configured
        if self.save_images:
            runner._save_scene_images(result, self.output_dir)
        
        # Return single image pair with SCARF stats
        rendered = result.rendered[0]  # [C, H, W]
        gt = result.ground_truth[0]  # [C, H, W]
        
        # Get cycles and fsdr_stats - always use simulation for now
        # (Real SCARF hooks integration would provide actual values)
        cycles, fsdr_stats = self._simulate_scarf_cycles()
        
        # Override with real FSDR stats if available
        if result.fsdr_stats:
            fsdr_stats = {
                'total_pixels': result.fsdr_stats.get('total_queries', 1000),
                'cache_hits': result.fsdr_stats.get('cache_hits', 680),
                'baseline_accesses': result.fsdr_stats.get('total_queries', 1000) * 32,
                'scarf_accesses': (result.fsdr_stats.get('cache_hits', 680) * 1 + 
                                   result.fsdr_stats.get('cache_misses', 320) * 32),
            }
        
        return rendered, gt, cycles, fsdr_stats
