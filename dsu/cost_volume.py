"""
Cost Volume

Feature matching cost computation.
"""
import torch
import torch.nn.functional as F

from .types import DSUConfig


class CostVolume:
    """
    Cost/correlation volume computation.
    
    Computes matching costs between reference and target features
    using cosine similarity (correlation) or L2 distance (cost).
    
    Hardware Mapping:
        - Normalization: divide by magnitude
        - Dot product: C parallel multiplies + adder tree
        - Total: ~500 LUTs, 32 DSPs (with reuse), 8 cycles
    
    Example:
        volume = CostVolume(config)
        costs = volume.compute_costs(ref_feature, target_features)
    """
    
    def __init__(self, config: DSUConfig):
        """Initialize cost volume."""
        self.config = config
        self.cost_type = config.cost_type
    
    def compute_costs(
        self,
        ref_feature: torch.Tensor,      # [C]
        target_features: torch.Tensor,  # [D, C]
    ) -> torch.Tensor:
        """
        Compute matching costs for all depth candidates.
        
        Args:
            ref_feature: Reference feature vector [C]
            target_features: Target features at each depth [D, C]
        
        Returns:
            [D] cost/correlation values
        """
        # Normalize features
        ref_norm = ref_feature / (torch.norm(ref_feature) + 1e-8)
        tgt_norms = target_features / (
            torch.norm(target_features, dim=1, keepdim=True) + 1e-8
        )
        
        # Cosine similarity via dot product
        costs = tgt_norms @ ref_norm  # [D]
        
        return costs
    
    def compute_single_cost(
        self,
        ref_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> float:
        """
        Compute single matching cost.
        
        Args:
            ref_feature: [C] reference feature
            target_feature: [C] target feature
        
        Returns:
            Cost/correlation value
        """
        ref_norm = ref_feature / (torch.norm(ref_feature) + 1e-8)
        tgt_norm = target_feature / (torch.norm(target_feature) + 1e-8)
        
        return float(torch.dot(ref_norm, tgt_norm))
    
    def compute_batch_costs(
        self,
        ref_features: torch.Tensor,      # [B, C]
        target_features: torch.Tensor,   # [B, D, C]
    ) -> torch.Tensor:
        """
        Batch cost computation.
        
        Args:
            ref_features: [B, C] batch of reference features
            target_features: [B, D, C] batch of target features
        
        Returns:
            [B, D] batch of costs
        """
        B, D, C = target_features.shape
        
        # Normalize
        ref_norm = F.normalize(ref_features, dim=1)  # [B, C]
        tgt_norm = F.normalize(target_features, dim=2)  # [B, D, C]
        
        # Batch dot product
        costs = torch.einsum('bc,bdc->bd', ref_norm, tgt_norm)
        
        return costs
