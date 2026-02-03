"""
Gaussian Merger

Enlarges and merges probe Gaussians for early-stop path,
scaling covariance to cover tile area while preserving appearance.
"""
import torch
from torch import Tensor
from typing import List, Tuple
import numpy as np

from .types import Gaussian


class GaussianMerger:
    """
    Enlarge and merge probe Gaussians to represent entire tile
    
    Strategy:
    - Position: Weighted average by opacity
    - Covariance: Average then scale up to cover tile area
    - Color: Weighted average of SH coefficients
    - Opacity: Average opacity
    
    This enables early-stop path to skip remaining pixels while maintaining
    reasonable rendering quality.
    """
    
    def __init__(self):
        """Initialize Gaussian merger"""
        pass
    
    def enlarge_and_merge(
        self,
        probe_gaussians: List[Gaussian],
        tile_coverage: Tuple[int, int, int, int],
    ) -> List[Gaussian]:
        """
        Enlarge probe Gaussians to cover entire tile
        
        Args:
            probe_gaussians: List of probe Gaussians (typically 4)
            tile_coverage: (row_start, row_end, col_start, col_end) in pixel coordinates
        
        Returns:
            List of enlarged Gaussians (typically 1 merged Gaussian)
        
        Strategy:
            1. Compute tile area from coverage
            2. Average probe Gaussian attributes (weighted by opacity)
            3. Scale covariance to cover tile area
            4. Return single enlarged Gaussian
        """
        if len(probe_gaussians) == 0:
            raise ValueError("Cannot merge empty Gaussian list")
        
        # Single Gaussian: just enlarge
        if len(probe_gaussians) == 1:
            return [self._enlarge_single_gaussian(probe_gaussians[0], tile_coverage)]
        
        # Multiple Gaussians: merge then enlarge
        merged = self._merge_gaussians(probe_gaussians)
        enlarged = self._enlarge_single_gaussian(merged, tile_coverage)
        
        return [enlarged]
    
    def _merge_gaussians(self, gaussians: List[Gaussian]) -> Gaussian:
        """
        Merge multiple Gaussians via weighted averaging
        
        Args:
            gaussians: List of Gaussians to merge
        
        Returns:
            Single merged Gaussian
        """
        # Extract attributes
        means = torch.stack([g.mean for g in gaussians])  # [N, 3]
        covs = torch.stack([g.cov for g in gaussians])  # [N, 3, 3]
        opacities = torch.tensor([g.opacity for g in gaussians])  # [N]
        harmonics = torch.stack([g.harmonics for g in gaussians])  # [N, C, D_sh]
        
        # Weighted average by opacity
        weights = opacities / opacities.sum()  # Normalize to sum=1
        weights = weights[:, None]  # [N, 1] for broadcasting
        
        # Compute weighted averages
        merged_mean = (means * weights).sum(dim=0)  # [3]
        merged_cov = (covs * weights[:, :, None]).sum(dim=0)  # [3, 3]
        merged_opacity = opacities.mean().item()  # scalar
        merged_harmonics = (harmonics * weights[:, :, None]).sum(dim=0)  # [C, D_sh]
        
        return Gaussian(
            mean=merged_mean,
            cov=merged_cov,
            opacity=merged_opacity,
            harmonics=merged_harmonics,
        )
    
    def _enlarge_single_gaussian(
        self,
        gaussian: Gaussian,
        tile_coverage: Tuple[int, int, int, int],
    ) -> Gaussian:
        """
        Enlarge a single Gaussian to cover tile area
        
        Args:
            gaussian: Gaussian to enlarge
            tile_coverage: (row_start, row_end, col_start, col_end)
        
        Returns:
            Enlarged Gaussian
        
        Strategy:
            Scale covariance proportionally to tile area to ensure coverage
        """
        # Compute tile area
        row_start, row_end, col_start, col_end = tile_coverage
        tile_height = row_end - row_start
        tile_width = col_end - col_start
        tile_area = tile_height * tile_width
        
        # Compute scale factor
        # Covariance scale should grow with sqrt(area) to maintain 2D coverage
        scale_factor = np.sqrt(tile_area) / 2.0  # Divide by 2 for conservative scaling
        
        # Scale covariance
        enlarged_cov = gaussian.cov * (scale_factor ** 2)
        
        # Position, color, opacity remain unchanged
        return Gaussian(
            mean=gaussian.mean.clone(),
            cov=enlarged_cov,
            opacity=gaussian.opacity,
            harmonics=gaussian.harmonics.clone(),
        )
    
    def _compute_tile_coverage_area(self, tile_coverage: Tuple[int, int, int, int]) -> int:
        """
        Compute tile area in pixels
        
        Args:
            tile_coverage: (row_start, row_end, col_start, col_end)
        
        Returns:
            Area in pixels
        """
        row_start, row_end, col_start, col_end = tile_coverage
        height = row_end - row_start
        width = col_end - col_start
        return height * width
    
    def _compute_average_position(self, gaussians: List[Gaussian]) -> Tensor:
        """
        Compute weighted average position
        
        Args:
            gaussians: List of Gaussians
        
        Returns:
            [3] - Average position
        """
        means = torch.stack([g.mean for g in gaussians])
        opacities = torch.tensor([g.opacity for g in gaussians])
        weights = opacities / opacities.sum()
        return (means * weights[:, None]).sum(dim=0)
    
    def _enlarge_covariance(self, cov: Tensor, scale_factor: float) -> Tensor:
        """
        Scale covariance matrix by factor
        
        Args:
            cov: [3, 3] - Original covariance
            scale_factor: Scaling factor
        
        Returns:
            [3, 3] - Enlarged covariance
        """
        return cov * (scale_factor ** 2)
    
    def _average_color(self, gaussians: List[Gaussian]) -> Tensor:
        """
        Average SH coefficients across Gaussians
        
        Args:
            gaussians: List of Gaussians
        
        Returns:
            [C, D_sh] - Averaged harmonics
        """
        harmonics = torch.stack([g.harmonics for g in gaussians])
        opacities = torch.tensor([g.opacity for g in gaussians])
        weights = opacities / opacities.sum()
        return (harmonics * weights[:, None, None]).sum(dim=0)
    
    def _average_opacity(self, gaussians: List[Gaussian]) -> float:
        """
        Average opacity across Gaussians
        
        Args:
            gaussians: List of Gaussians
        
        Returns:
            Average opacity
        """
        opacities = [g.opacity for g in gaussians]
        return float(np.mean(opacities))
    
    def _create_enlarged_gaussian(
        self,
        position: Tensor,
        covariance: Tensor,
        opacity: float,
        harmonics: Tensor,
    ) -> Gaussian:
        """
        Create enlarged Gaussian from averaged attributes
        
        Args:
            position: [3] - Averaged position
            covariance: [3, 3] - Enlarged covariance
            opacity: Averaged opacity
            harmonics: [C, D_sh] - Averaged SH coefficients
        
        Returns:
            Enlarged Gaussian
        """
        return Gaussian(
            mean=position,
            cov=covariance,
            opacity=opacity,
            harmonics=harmonics,
        )
