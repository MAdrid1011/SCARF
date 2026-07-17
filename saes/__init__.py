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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .progressive_saes import ProgressiveSAES, apply_progressive_saes

__version__ = "0.1.0"

__all__ = [
    # Progressive SAES v3 (Multi-Level ASIC model)
    "ProgressiveSAES",
    "apply_progressive_saes",
]


def __getattr__(name: str) -> Any:
    """Load the Torch implementation only for callers that request it.

    Result-schema and accounting utilities live under this package too, but
    run in the lightweight aggregation interpreter where Torch is optional.
    """
    if name in {"ProgressiveSAES", "apply_progressive_saes"}:
        from .progressive_saes import ProgressiveSAES, apply_progressive_saes

        return {
            "ProgressiveSAES": ProgressiveSAES,
            "apply_progressive_saes": apply_progressive_saes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
