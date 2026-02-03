"""
Cycle Counter

Hardware cycle counting infrastructure for SCARF benchmarks.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


# Hardware cycle constants based on architecture documentation
CYCLE_CONSTANTS = {
    # FSDR cycles
    'fsdr.hash': 3,
    'fsdr.lookup': 5,
    'fsdr.direct_reuse': 1,
    'fsdr.interpolation': 4,
    'fsdr.light_verify_base': 3,  # Plus depth count
    
    # DSU cycles
    'dsu.project': 8,
    'dsu.sample': 4,
    'dsu.cost': 2,
    'dsu.softmax': 6,
    
    # GGU cycles
    'ggu.position': 2,
    'ggu.covariance': 5,
    'ggu.sh_rotation': 3,
    
    # SAES cycles
    'saes.probe': 4,
    'saes.similarity': 6,
    'saes.decision': 1,
    'saes.merge': 4,
    
    # Encoder cycles (per element/operation)
    'encoder.conv_mac': 1,       # Per MAC in systolic array
    'encoder.conv_setup': 10,    # Convolution setup overhead
    'encoder.gemm_tile': 10,     # GEMM tile overhead
    'encoder.activation': 1,     # Per element activation (LUT-based)
    'encoder.norm_mean': 1,      # Per element mean accumulation
    'encoder.norm_var': 1,       # Per element var accumulation
    'encoder.norm_apply': 1,     # Per element normalize
    'encoder.norm_overhead': 2,  # Division/sqrt overhead
    'encoder.bilinear_coord': 1, # Coordinate calculation
    'encoder.bilinear_sample': 1,# 4-point sampling
    'encoder.bilinear_interp': 2,# Interpolation (4 MACs)
}


@dataclass
class CycleEstimate:
    """
    Single cycle estimate for an operation.
    
    Attributes:
        component: Component name ('fsdr', 'dsu', 'ggu', 'saes')
        operation: Operation name ('hash', 'lookup', 'project', etc.)
        cycles: Estimated hardware cycles
        memory_accesses: Number of memory accesses
    """
    component: str
    operation: str
    cycles: int
    memory_accesses: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'component': self.component,
            'operation': self.operation,
            'cycles': self.cycles,
            'memory_accesses': self.memory_accesses,
        }


@dataclass
class CycleSummary:
    """
    Summary of cycles for a component.
    
    Attributes:
        component: Component name
        total_cycles: Total cycles for this component
        operation_breakdown: Cycles per operation
        memory_accesses: Total memory accesses
        percentage: Percentage of total cycles
    """
    component: str
    total_cycles: int
    operation_breakdown: Dict[str, int]
    memory_accesses: int
    percentage: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'component': self.component,
            'total_cycles': self.total_cycles,
            'operation_breakdown': self.operation_breakdown,
            'memory_accesses': self.memory_accesses,
            'percentage': self.percentage,
        }


class CycleCounter:
    """
    Hardware cycle counter for SCARF benchmark.
    
    Tracks cycle counts for each operation across components.
    
    Example:
        counter = CycleCounter()
        counter.record('fsdr', 'hash', 3)
        counter.record('fsdr', 'lookup', 5, memory_accesses=1)
        counter.record('dsu', 'project', 8)
        
        print(f"Total: {counter.get_total_cycles()} cycles")
        print(f"Memory: {counter.get_memory_accesses()} accesses")
        
        summary = counter.get_summary()
        for component, stats in summary.items():
            print(f"{component}: {stats.total_cycles} cycles ({stats.percentage:.1f}%)")
    """
    
    def __init__(self, clock_freq_mhz: int = 200):
        """
        Initialize cycle counter.
        
        Args:
            clock_freq_mhz: Clock frequency in MHz for timing conversion
        """
        self.clock_freq_mhz = clock_freq_mhz
        self._records: List[CycleEstimate] = []
        self._by_component: Dict[str, List[CycleEstimate]] = defaultdict(list)
    
    def record(
        self,
        component: str,
        operation: str,
        cycles: int,
        memory_accesses: int = 0,
    ):
        """
        Record a cycle estimate.
        
        Args:
            component: Component name ('fsdr', 'dsu', 'ggu', 'saes')
            operation: Operation name ('hash', 'lookup', etc.)
            cycles: Number of cycles
            memory_accesses: Number of memory accesses
        """
        estimate = CycleEstimate(
            component=component,
            operation=operation,
            cycles=cycles,
            memory_accesses=memory_accesses,
        )
        self._records.append(estimate)
        self._by_component[component].append(estimate)
    
    def record_from_constant(
        self,
        key: str,
        memory_accesses: int = 0,
    ):
        """
        Record using predefined constant.
        
        Args:
            key: Constant key (e.g., 'fsdr.hash', 'dsu.project')
            memory_accesses: Number of memory accesses
        """
        if key not in CYCLE_CONSTANTS:
            raise ValueError(f"Unknown cycle constant: {key}")
        
        component, operation = key.split('.')
        cycles = CYCLE_CONSTANTS[key]
        self.record(component, operation, cycles, memory_accesses)
    
    def get_total_cycles(self) -> int:
        """Get total cycle count."""
        return sum(r.cycles for r in self._records)
    
    def get_memory_accesses(self) -> int:
        """Get total memory accesses."""
        return sum(r.memory_accesses for r in self._records)
    
    def get_summary(self) -> Dict[str, CycleSummary]:
        """
        Get summary by component.
        
        Returns:
            Dict mapping component name to CycleSummary
        """
        total = self.get_total_cycles()
        summary = {}
        
        for component, records in self._by_component.items():
            # Aggregate operations
            operation_breakdown = defaultdict(int)
            total_cycles = 0
            memory_accesses = 0
            
            for r in records:
                operation_breakdown[r.operation] += r.cycles
                total_cycles += r.cycles
                memory_accesses += r.memory_accesses
            
            percentage = (total_cycles / total * 100) if total > 0 else 0
            
            summary[component] = CycleSummary(
                component=component,
                total_cycles=total_cycles,
                operation_breakdown=dict(operation_breakdown),
                memory_accesses=memory_accesses,
                percentage=percentage,
            )
        
        return summary
    
    def get_time_us(self) -> float:
        """Get estimated time in microseconds."""
        cycles = self.get_total_cycles()
        return cycles / self.clock_freq_mhz
    
    def reset(self):
        """Clear all recorded cycles."""
        self._records.clear()
        self._by_component.clear()
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        summary = self.get_summary()
        return {
            'total_cycles': self.get_total_cycles(),
            'total_memory_accesses': self.get_memory_accesses(),
            'time_us': self.get_time_us(),
            'by_component': {k: v.to_dict() for k, v in summary.items()},
        }
