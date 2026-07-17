"""
Integration Module

Model loading and data preparation for SCARF accelerator.

Example:
    from integration import create_model_loader

    loader = create_model_loader('transplat')
    model_bundle = loader.load_model('checkpoints/re10k.ckpt')
    data_bundle = loader.load_data(model_bundle)
"""

from .model_loader import (
    BaseModelLoader,
    ModelBundle,
    DataBundle,
    TransplatLoader,
    MVSplatLoader,
    DepthSplatLoader,
    create_model_loader,
    load_target_free_calibration_data,
)

__all__ = [
    'BaseModelLoader',
    'ModelBundle',
    'DataBundle',
    'TransplatLoader',
    'MVSplatLoader',
    'DepthSplatLoader',
    'create_model_loader',
    'load_target_free_calibration_data',
]
