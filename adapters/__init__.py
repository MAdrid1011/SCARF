"""
Model Adapters Module

Adapters for different generalizable 3DGS models.

Supported Models:
- Transplat: Cost volume with negative softmax, inverse depth
- MVSplat: Correlation volume with direct softmax, linear depth
- DepthSplat: DINOv2 features, 3-view support

Example:
    from adapters import create_adapter
    
    adapter = create_adapter('transplat')
    probs = adapter.extract_depth_distribution(cost_volume, depth_candidates)
"""

from .base_adapter import BaseAdapter
from .transplat_adapter import TransplatAdapter
from .mvsplat_adapter import MVSplatAdapter
from .depthsplat_adapter import DepthSplatAdapter


def create_adapter(model_type: str) -> BaseAdapter:
    """
    Factory function to create appropriate adapter.
    
    Args:
        model_type: 'transplat', 'mvsplat', or 'depthsplat'
    
    Returns:
        Configured adapter instance
    """
    adapters = {
        'transplat': TransplatAdapter,
        'mvsplat': MVSplatAdapter,
        'depthsplat': DepthSplatAdapter,
    }
    
    if model_type not in adapters:
        raise ValueError(
            f"Unknown model: {model_type}. "
            f"Supported: {list(adapters.keys())}"
        )
    
    return adapters[model_type]()


__all__ = [
    'BaseAdapter',
    'TransplatAdapter',
    'MVSplatAdapter', 
    'DepthSplatAdapter',
    'create_adapter',
]
