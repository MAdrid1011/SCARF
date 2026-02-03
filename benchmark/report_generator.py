"""
Report Generator

Benchmark report generation for SCARF.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .cycle_counter import CycleCounter, CycleSummary
from .quality_validator import QualityMetrics


@dataclass
class BenchmarkReport:
    """
    Single benchmark run report.
    
    Attributes:
        name: Benchmark name ('baseline' or 'scarf')
        quality: Quality metrics
        cycles: Cycle counter (optional, for SCARF)
        fsdr_stats: FSDR-specific stats (optional)
        num_scenes: Number of scenes processed
    """
    name: str
    quality: QualityMetrics
    cycles: Optional[Dict] = None
    fsdr_stats: Optional[Dict] = None
    num_scenes: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            'name': self.name,
            'quality': self.quality.to_dict(),
            'num_scenes': self.num_scenes,
        }
        if self.cycles:
            result['cycles'] = self.cycles
        if self.fsdr_stats:
            result['fsdr_stats'] = self.fsdr_stats
        return result


@dataclass
class ComparisonReport:
    """
    Comparison between baseline and SCARF runs.
    
    Attributes:
        baseline: Baseline report
        scarf: SCARF report
        speedup: Cycle speedup factor
        psnr_delta: PSNR difference
        ssim_delta: SSIM difference
        quality_acceptable: Whether quality meets threshold
        metadata: Additional metadata
    """
    baseline: BenchmarkReport
    scarf: BenchmarkReport
    speedup: float = 0.0
    psnr_delta: float = 0.0
    ssim_delta: float = 0.0
    quality_acceptable: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'metadata': self.metadata,
            'baseline': self.baseline.to_dict(),
            'scarf': self.scarf.to_dict(),
            'comparison': {
                'speedup': self.speedup,
                'psnr_delta': self.psnr_delta,
                'ssim_delta': self.ssim_delta,
                'quality_acceptable': self.quality_acceptable,
            },
        }


class ReportGenerator:
    """
    Generate benchmark reports.
    
    Example:
        generator = ReportGenerator()
        
        baseline = generator.generate_baseline_report(
            quality=baseline_metrics,
            num_scenes=100,
        )
        
        scarf = generator.generate_scarf_report(
            quality=scarf_metrics,
            cycles=cycle_counter,
            fsdr_stats={'hit_rate': 0.68},
            num_scenes=100,
        )
        
        comparison = generator.generate_comparison(baseline, scarf)
        
        generator.save_json(comparison, 'report.json')
        print(generator.to_markdown(comparison))
    """
    
    def __init__(self, model_name: str = 'transplat'):
        """
        Initialize report generator.
        
        Args:
            model_name: Model name for metadata
        """
        self.model_name = model_name
    
    def generate_baseline_report(
        self,
        quality: QualityMetrics,
        num_scenes: int,
    ) -> BenchmarkReport:
        """Generate baseline benchmark report."""
        return BenchmarkReport(
            name='baseline',
            quality=quality,
            num_scenes=num_scenes,
        )
    
    def generate_scarf_report(
        self,
        quality: QualityMetrics,
        cycles: CycleCounter,
        fsdr_stats: Optional[Dict] = None,
        num_scenes: int = 0,
    ) -> BenchmarkReport:
        """Generate SCARF benchmark report."""
        return BenchmarkReport(
            name='scarf',
            quality=quality,
            cycles=cycles.to_dict() if cycles else None,
            fsdr_stats=fsdr_stats,
            num_scenes=num_scenes,
        )
    
    def generate_comparison(
        self,
        baseline: BenchmarkReport,
        scarf: BenchmarkReport,
        baseline_cycles_per_pixel: int = 454,
        psnr_threshold: float = 0.5,
        ssim_threshold: float = 0.01,
    ) -> ComparisonReport:
        """
        Generate comparison report.
        
        Args:
            baseline: Baseline report
            scarf: SCARF report
            baseline_cycles_per_pixel: Baseline cycle count for speedup calculation
            psnr_threshold: Maximum acceptable PSNR drop
            ssim_threshold: Maximum acceptable SSIM drop
        """
        # Calculate deltas
        psnr_delta = scarf.quality.psnr_mean - baseline.quality.psnr_mean
        ssim_delta = scarf.quality.ssim_mean - baseline.quality.ssim_mean
        
        # Calculate speedup based on total cycles
        speedup = 1.0
        if scarf.cycles and scarf.fsdr_stats:
            total_scarf_cycles = scarf.cycles.get('total_cycles', 0)
            total_pixels = scarf.fsdr_stats.get('total_pixels', 0)
            
            if total_pixels > 0 and total_scarf_cycles > 0:
                # Total baseline cycles for same pixels
                total_baseline_cycles = baseline_cycles_per_pixel * total_pixels
                speedup = total_baseline_cycles / total_scarf_cycles
        
        # Check quality threshold
        quality_acceptable = (
            abs(psnr_delta) < psnr_threshold and
            abs(ssim_delta) < ssim_threshold
        )
        
        # Build metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'num_scenes': baseline.num_scenes,
        }
        
        return ComparisonReport(
            baseline=baseline,
            scarf=scarf,
            speedup=speedup,
            psnr_delta=psnr_delta,
            ssim_delta=ssim_delta,
            quality_acceptable=quality_acceptable,
            metadata=metadata,
        )
    
    def to_json(self, report: ComparisonReport) -> str:
        """Convert report to JSON string."""
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        return json.dumps(recursive_convert(report.to_dict()), indent=2)
    
    def save_json(self, report: ComparisonReport, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json(report))
    
    def to_markdown(self, report: ComparisonReport) -> str:
        """Convert report to markdown summary."""
        lines = [
            "# SCARF Benchmark Report",
            "",
            f"**Model:** {report.metadata.get('model', 'unknown')}",
            f"**Timestamp:** {report.metadata.get('timestamp', 'unknown')}",
            f"**Scenes:** {report.metadata.get('num_scenes', 0)}",
            "",
            "## Performance",
            "",
            f"| Metric | Baseline | SCARF | Delta |",
            f"|--------|----------|-------|-------|",
            f"| PSNR (dB) | {report.baseline.quality.psnr_mean:.2f} | "
            f"{report.scarf.quality.psnr_mean:.2f} | {report.psnr_delta:+.2f} |",
            f"| SSIM | {report.baseline.quality.ssim_mean:.4f} | "
            f"{report.scarf.quality.ssim_mean:.4f} | {report.ssim_delta:+.4f} |",
            "",
            f"**Speedup:** {report.speedup:.2f}×",
            f"**Quality Acceptable:** {'Yes' if report.quality_acceptable else 'No'}",
            "",
        ]
        
        # Add FSDR stats if available
        if report.scarf.fsdr_stats:
            lines.extend([
                "## FSDR Statistics",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
            ])
            for key, value in report.scarf.fsdr_stats.items():
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.2%} |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")
        
        # Add cycle breakdown if available
        if report.scarf.cycles:
            lines.extend([
                "## Cycle Breakdown",
                "",
                f"**Total Cycles:** {report.scarf.cycles.get('total_cycles', 0)}",
                f"**Memory Accesses:** {report.scarf.cycles.get('total_memory_accesses', 0)}",
                "",
            ])
            
            by_component = report.scarf.cycles.get('by_component', {})
            if by_component:
                lines.extend([
                    "| Component | Cycles | % |",
                    "|-----------|--------|---|",
                ])
                for comp, stats in by_component.items():
                    lines.append(
                        f"| {comp} | {stats['total_cycles']} | "
                        f"{stats['percentage']:.1f}% |"
                    )
                lines.append("")
        
        return "\n".join(lines)
