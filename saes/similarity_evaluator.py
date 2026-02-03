"""
3D Gaussian Similarity Evaluator

Computes similarity metrics among probe Gaussians based on:
- Position dispersion (spatial spread)
- Covariance dispersion (shape variation)
- Color dispersion (SH coefficient differences)
- Opacity dispersion (transparency variation)

Weighted aggregation produces similarity score in [0, 1].
"""
import torch
from torch import Tensor
from typing import List, Optional
import numpy as np

from .types import Gaussian, GaussianSimilarityMetrics


class GaussianSimilarityEvaluator:
    """
    Evaluates 3D Gaussian similarity for probe Gaussians
    
    The similarity score indicates how similar probe Gaussians are in 3D space,
    guiding the decision whether to early-stop or continue processing.
    
    Weights (from Design2 Section 5.1):
    - Position: 0.4 (most important - spatial continuity)
    - Covariance: 0.3 (shape consistency)
    - Color: 0.15 (appearance)
    - Opacity: 0.15 (transparency)
    """
    
    def __init__(self, scene_scale: Optional[float] = None):
        """
        Initialize evaluator
        
        Args:
            scene_scale: Optional scene scale for position normalization.
                        If None, will auto-estimate from Gaussian positions.
        """
        self.scene_scale = scene_scale
    
    def evaluate(self, gaussians: List[Gaussian]) -> GaussianSimilarityMetrics:
        """
        Compute similarity metrics among probe Gaussians
        
        Args:
            gaussians: List of probe Gaussians (typically 4)
        
        Returns:
            GaussianSimilarityMetrics with dispersion values and similarity score
        """
        if len(gaussians) == 0:
            raise ValueError("Cannot evaluate similarity of empty Gaussian list")
        
        # Single Gaussian has perfect similarity
        if len(gaussians) == 1:
            return GaussianSimilarityMetrics(
                position_dispersion=0.0,
                covariance_dispersion=0.0,
                color_dispersion=0.0,
                opacity_dispersion=0.0,
                similarity_score=1.0,
            )
        
        # Extract attributes
        positions = torch.stack([g.mean for g in gaussians])  # [N, 3]
        covariances = torch.stack([g.cov for g in gaussians])  # [N, 3, 3]
        opacities = torch.tensor([g.opacity for g in gaussians])  # [N]
        harmonics = torch.stack([g.harmonics for g in gaussians])  # [N, C, D_sh]
        
        # Compute dispersions
        pos_disp = self._compute_position_dispersion(positions)
        cov_disp = self._compute_covariance_dispersion(covariances)
        color_disp = self._compute_color_dispersion(harmonics)
        opacity_disp = self._compute_opacity_dispersion(opacities)
        
        # Compute similarity score
        similarity = self._compute_similarity_score(pos_disp, cov_disp, color_disp, opacity_disp)
        
        return GaussianSimilarityMetrics(
            position_dispersion=pos_disp,
            covariance_dispersion=cov_disp,
            color_dispersion=color_disp,
            opacity_dispersion=opacity_disp,
            similarity_score=similarity,
        )
    
    def _compute_position_dispersion(self, positions: Tensor) -> float:
        """
        Compute position dispersion: max pairwise distance / scene_scale
        
        Args:
            positions: [N, 3] - Gaussian positions
        
        Returns:
            Normalized position dispersion
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(positions)  # [N, N]
        
        # Max pairwise distance
        max_dist = distances.max().item()
        
        # Normalize by scene scale
        if self.scene_scale is None:
            # Auto-estimate from positions
            self.scene_scale = self._estimate_scene_scale(positions)
        
        # Avoid division by zero
        scene_scale = max(self.scene_scale, 1e-6)
        
        return max_dist / scene_scale
    
    def _compute_covariance_dispersion(self, covariances: Tensor) -> float:
        """
        Compute covariance dispersion: max pairwise Frobenius norm / avg_cov_norm
        
        Args:
            covariances: [N, 3, 3] - Covariance matrices
        
        Returns:
            Normalized covariance dispersion
        """
        # Flatten covariances for pairwise distance computation
        cov_flat = covariances.reshape(covariances.shape[0], -1)  # [N, 9]
        
        # Compute pairwise Frobenius norm distances
        distances = self._pairwise_distances(cov_flat)  # [N, N]
        
        # Max pairwise distance
        max_dist = distances.max().item()
        
        # Compute average covariance norm for normalization
        cov_norms = torch.norm(covariances, p='fro', dim=(-2, -1))  # [N]
        avg_cov_norm = cov_norms.mean().item()
        
        # Avoid division by zero
        avg_cov_norm = max(avg_cov_norm, 1e-6)
        
        return max_dist / avg_cov_norm
    
    def _compute_color_dispersion(self, harmonics: Tensor) -> float:
        """
        Compute color dispersion: max pairwise SH DC component distance
        
        Args:
            harmonics: [N, C, D_sh] - Spherical harmonics coefficients
        
        Returns:
            Color dispersion (RGB from first 3 SH DC coefficients)
        """
        # Extract DC component (first SH coefficient represents base color)
        # Assuming harmonics[:, :, 0] or harmonics[:, :3, 0] contains RGB DC
        if harmonics.shape[1] >= 3:
            # Use first 3 channels as RGB
            sh_dc = harmonics[:, :3, 0]  # [N, 3]
        else:
            # Fall back to using all channels
            sh_dc = harmonics[:, :, 0]  # [N, C]
        
        # Compute pairwise distances
        distances = self._pairwise_distances(sh_dc)  # [N, N]
        
        # Max pairwise distance
        max_dist = distances.max().item()
        
        return max_dist
    
    def _compute_opacity_dispersion(self, opacities: Tensor) -> float:
        """
        Compute opacity dispersion: max - min
        
        Args:
            opacities: [N] - Opacity values
        
        Returns:
            Opacity dispersion
        """
        return (opacities.max() - opacities.min()).item()
    
    def _compute_similarity_score(
        self,
        pos_disp: float,
        cov_disp: float,
        color_disp: float,
        opacity_disp: float,
    ) -> float:
        """
        Compute weighted similarity score from dispersions
        
        Formula (from Design2):
            dispersion = 0.4*pos + 0.3*cov + 0.15*color + 0.15*opacity
            similarity = exp(-dispersion / 0.1)
        
        Returns:
            Similarity score in [0, 1], where 1.0 = perfect similarity
        """
        # Weighted aggregation
        weighted_dispersion = (
            0.4 * pos_disp +
            0.3 * cov_disp +
            0.15 * color_disp +
            0.15 * opacity_disp
        )
        
        # Exponential mapping to [0, 1]
        # Temperature = 0.1 controls sensitivity
        similarity = np.exp(-weighted_dispersion / 0.1)
        
        # Clamp to [0, 1] for numerical stability
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def _pairwise_distances(self, vectors: Tensor) -> Tensor:
        """
        Compute all pairwise L2 distances efficiently
        
        Args:
            vectors: [N, D] - Vector collection
        
        Returns:
            distances: [N, N] - where distances[i,j] = ||vectors[i] - vectors[j]||
        
        Implementation:
            ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
        """
        # Compute dot products
        dots = vectors @ vectors.T  # [N, N]
        
        # Compute squared norms
        norms_sq = torch.diag(dots)  # [N]
        
        # Compute squared distances
        # distances[i,j]² = norms[i] + norms[j] - 2*dots[i,j]
        distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        
        # Take sqrt and clamp for numerical stability
        distances = torch.sqrt(torch.clamp(distances_sq, min=0.0))
        
        return distances
    
    def _estimate_scene_scale(self, positions: Tensor) -> float:
        """
        Auto-estimate scene scale from Gaussian positions
        
        Args:
            positions: [N, 3] - Gaussian positions
        
        Returns:
            scene_scale: Standard deviation of positions
        
        Rationale:
            Scene scale represents typical spatial extent of the scene.
            Using std normalizes position dispersion to be comparable across scenes.
        """
        return positions.std().item()
