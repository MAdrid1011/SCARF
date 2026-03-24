"""
SAES: Scene-Adaptive Early Sparsification Dataflow

Multi-level early sparsification for tile-based feedback processing
in generalizable 3DGS encoders.

Key Component:
- ProgressiveSAES: Multi-level (L0/L1/L2) early-stopping ASIC model

Example:
    from saes import ProgressiveSAES, apply_progressive_saes

    saes = ProgressiveSAES(config)
    result = apply_progressive_saes(saes, gaussians, depths, features)
"""

from .progressive_saes import ProgressiveSAES, apply_progressive_saes

__version__ = "0.1.0"

__all__ = [
    # Progressive SAES v3 (Multi-Level ASIC model)
    "ProgressiveSAES",
    "apply_progressive_saes",
]
