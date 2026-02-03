# CycleCounter

Hardware cycle counting infrastructure for SCARF benchmarks.

## Overview

Tracks estimated hardware cycles for each operation across SCARF components (FSDR, DSU, GGU, SAES), enabling performance analysis and speedup calculations.

## External Interface

### CYCLE_CONSTANTS

Predefined hardware cycle constants based on architecture documentation:

```python
CYCLE_CONSTANTS = {
    # FSDR cycles
    'fsdr.hash': 3,           # LSH signature generation
    'fsdr.lookup': 5,         # Cache lookup
    'fsdr.direct_reuse': 1,   # Pass-through
    'fsdr.interpolation': 4,  # Weighted blend
    'fsdr.light_verify_base': 3,  # Base cycles for light verification
    
    # DSU cycles
    'dsu.project': 8,         # 3D-2D projection
    'dsu.sample': 4,          # Bilinear sampling
    'dsu.cost': 2,            # Cost computation
    'dsu.softmax': 6,         # Softmax aggregation
    
    # GGU cycles
    'ggu.position': 2,        # Position calculation
    'ggu.covariance': 5,      # Covariance building
    'ggu.sh_rotation': 3,     # SH rotation
    
    # SAES cycles
    'saes.probe': 4,          # Probe sampling
    'saes.similarity': 6,     # Similarity computation
    'saes.decision': 1,       # Decision logic
    'saes.merge': 4,          # Gaussian merging
}
```

### CycleEstimate

```python
@dataclass
class CycleEstimate:
    component: str       # 'fsdr', 'dsu', 'ggu', 'saes'
    operation: str       # Operation name
    cycles: int          # Estimated cycles
    memory_accesses: int # Number of memory accesses (default: 0)
    
    def to_dict(self) -> Dict
```

### CycleSummary

```python
@dataclass
class CycleSummary:
    component: str                    # Component name
    total_cycles: int                 # Total cycles for component
    operation_breakdown: Dict[str, int]  # Cycles per operation
    memory_accesses: int              # Total memory accesses
    percentage: float                 # Percentage of total cycles
    
    def to_dict(self) -> Dict
```

### CycleCounter

```python
class CycleCounter:
    def __init__(self, clock_freq_mhz: int = 200)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `record(component, operation, cycles, memory_accesses=0)` | Record a cycle estimate |
| `record_from_constant(key, memory_accesses=0)` | Record using predefined constant |
| `get_total_cycles()` | Get total cycle count |
| `get_memory_accesses()` | Get total memory accesses |
| `get_summary()` | Get summary by component |
| `get_time_us()` | Get estimated time in microseconds |
| `reset()` | Clear all recorded cycles |
| `to_dict()` | Export as dictionary |

## Usage Example

```python
from benchmark import CycleCounter, CYCLE_CONSTANTS

counter = CycleCounter(clock_freq_mhz=200)

# Record operations
counter.record('fsdr', 'hash', 3)
counter.record('fsdr', 'lookup', 5, memory_accesses=1)
counter.record('dsu', 'project', 8)

# Or use predefined constants
counter.record_from_constant('fsdr.hash')

# Get statistics
print(f"Total: {counter.get_total_cycles()} cycles")
print(f"Memory: {counter.get_memory_accesses()} accesses")
print(f"Time: {counter.get_time_us():.2f} µs")

# Get breakdown
summary = counter.get_summary()
for component, stats in summary.items():
    print(f"{component}: {stats.total_cycles} cycles ({stats.percentage:.1f}%)")
```

## Hardware Mapping

The cycle constants map to hardware resources:

| Operation | Cycles | Hardware Resources |
|-----------|--------|-------------------|
| fsdr.hash | 3 | 16 comparators |
| fsdr.lookup | 5 | 256 Hamming comparators |
| dsu.project | 8 | 8 DSPs |
| dsu.sample | 4 | 8 DSPs |
| ggu.covariance | 5 | 12 DSPs |

## Related Files

- `benchmark_runner.py` - Uses CycleCounter for SCARF runs
- `fsdr/fsdr_processor.py` - FSDR cycle counting integration
- `dsu/dsu_processor.py` - DSU cycle counting integration
- `ggu/ggu_processor.py` - GGU cycle counting integration
