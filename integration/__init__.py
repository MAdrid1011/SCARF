"""
Integration Module

Complete SCARF accelerator integrating all components.

Example:
    from integration import Accelerator, create_model_loader
    
    # Load model
    loader = create_model_loader('transplat')
    model_bundle = loader.load_model('checkpoints/re10k.ckpt')
    data_bundle = loader.load_data(model_bundle)
    
    # Run accelerator
    accelerator = Accelerator(model_type='transplat')
    gaussians, profiling = accelerator.process_scene(
        features, intrinsics, extrinsics, near, far
    )
"""

from .accelerator import Accelerator
from .pipeline import Pipeline
from .model_loader import (
    BaseModelLoader,
    ModelBundle,
    DataBundle,
    TransplatLoader,
    MVSplatLoader,
    DepthSplatLoader,
    create_model_loader,
)

__all__ = [
    'Accelerator',
    'Pipeline',
    'BaseModelLoader',
    'ModelBundle',
    'DataBundle',
    'TransplatLoader',
    'MVSplatLoader',
    'DepthSplatLoader',
    'create_model_loader',
]
