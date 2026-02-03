"""
Integration Module

Complete SCARF accelerator integrating all components.

Example:
    from integration import Accelerator
    
    accelerator = Accelerator(model_type='transplat')
    gaussians, profiling = accelerator.process_scene(
        features, intrinsics, extrinsics, near, far
    )
"""

from .accelerator import Accelerator
from .pipeline import Pipeline

__all__ = ['Accelerator', 'Pipeline']
