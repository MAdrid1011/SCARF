"""
Benchmark Module

Performance and quality benchmarking for SCARF accelerator.

Example:
    from benchmark import CycleCounter, QualityValidator, BenchmarkRunner
    
    # Count cycles
    counter = CycleCounter()
    counter.record('fsdr', 'hash', 3)
    print(counter.get_total_cycles())
    
    # Validate quality
    validator = QualityValidator()
    validator.record_sample(rendered, ground_truth)
    print(validator.get_metrics())
"""

from .cycle_counter import (
    CycleEstimate,
    CycleSummary,
    CycleCounter,
    CYCLE_CONSTANTS,
)
from .quality_validator import (
    QualityMetrics,
    QualityValidator,
)
from .report_generator import (
    BenchmarkReport,
    ComparisonReport,
    ReportGenerator,
)
from .benchmark_runner import (
    BaselineResult,
    SCARFResult,
    BenchmarkRunner,
)


__all__ = [
    # Cycle counting
    'CycleEstimate',
    'CycleSummary',
    'CycleCounter',
    'CYCLE_CONSTANTS',
    # Quality validation
    'QualityMetrics',
    'QualityValidator',
    # Report generation
    'BenchmarkReport',
    'ComparisonReport',
    'ReportGenerator',
    # Benchmark runner
    'BaselineResult',
    'SCARFResult',
    'BenchmarkRunner',
]
