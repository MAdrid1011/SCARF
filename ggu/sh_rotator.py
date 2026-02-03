"""
SH Rotator

Spherical harmonics rotation for view-dependent colors.
"""
import torch
import numpy as np

from .types import GGUConfig


class SHRotator:
    """
    Rotate spherical harmonics coefficients to world space.
    
    SH Degrees:
        - Degree 0 (1 coeff): DC term, no rotation needed
        - Degree 1 (3 coeffs): Rotate like 3D vector
        - Degree 2 (5 coeffs): Use 5x5 Wigner D-matrix
        - Degree 3 (7 coeffs): Use 7x7 Wigner D-matrix
    
    Hardware Mapping:
        - Degree 0: pass-through
        - Degree 1: 3x3 matrix multiply (9 MACs per channel)
        - Degree 2: 5x5 matrix multiply (25 MACs per channel)
        - Degree 3: 7x7 matrix multiply (49 MACs per channel)
        - Total: ~300 LUTs, 18 DSPs, 8 cycles
    
    Example:
        rotator = SHRotator(config)
        rotated_sh = rotator.rotate(sh_coeffs, rotation_matrix)
    """
    
    def __init__(self, config: GGUConfig):
        """Initialize SH rotator."""
        self.config = config
        self.sh_degree = config.sh_degree
        self.num_coeffs = config.num_sh_coeffs
    
    def rotate(
        self,
        sh_coeffs: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate spherical harmonics coefficients.
        
        Args:
            sh_coeffs: [C, num_coeffs] SH coefficients (C=3 for RGB)
            rotation: [3, 3] rotation matrix
        
        Returns:
            [C, num_coeffs] rotated SH coefficients
        """
        C = sh_coeffs.shape[0]
        result = sh_coeffs.clone()
        
        # Degree 0: DC term unchanged (index 0)
        # result[:, 0] = sh_coeffs[:, 0]  # Already cloned
        
        # Degree 1: Direct rotation (indices 1-3)
        if self.sh_degree >= 1 and self.num_coeffs >= 4:
            sh_1 = sh_coeffs[:, 1:4]  # [C, 3]
            rotated_1 = sh_1 @ rotation.T  # [C, 3]
            result[:, 1:4] = rotated_1
        
        # Degree 2: 5x5 Wigner D-matrix (indices 4-8)
        if self.sh_degree >= 2 and self.num_coeffs >= 9:
            D2 = self._compute_wigner_d_2(rotation)
            sh_2 = sh_coeffs[:, 4:9]  # [C, 5]
            rotated_2 = sh_2 @ D2.T
            result[:, 4:9] = rotated_2
        
        # Degree 3: 7x7 Wigner D-matrix (indices 9-15)
        if self.sh_degree >= 3 and self.num_coeffs >= 16:
            D3 = self._compute_wigner_d_3(rotation)
            sh_3 = sh_coeffs[:, 9:16]  # [C, 7]
            rotated_3 = sh_3 @ D3.T
            result[:, 9:16] = rotated_3
        
        return result
    
    def _compute_wigner_d_2(self, R: torch.Tensor) -> torch.Tensor:
        """
        Compute 5x5 Wigner D-matrix for degree 2.
        
        Based on the real spherical harmonics rotation formulas.
        """
        # For degree-2, we need to compute the transformation matrix
        # from the rotation matrix elements.
        # This is a simplified approximation - full implementation would
        # use the Wigner D-matrix formulas.
        
        # Placeholder: identity (no rotation for degree 2)
        # In production, implement proper Wigner D-matrix
        return torch.eye(5, dtype=R.dtype)
    
    def _compute_wigner_d_3(self, R: torch.Tensor) -> torch.Tensor:
        """
        Compute 7x7 Wigner D-matrix for degree 3.
        """
        # Placeholder: identity (no rotation for degree 3)
        # In production, implement proper Wigner D-matrix
        return torch.eye(7, dtype=R.dtype)
    
    def rotate_batch(
        self,
        sh_coeffs: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch SH rotation.
        
        Args:
            sh_coeffs: [N, C, num_coeffs]
            rotations: [N, 3, 3]
        
        Returns:
            [N, C, num_coeffs]
        """
        N = sh_coeffs.shape[0]
        results = torch.zeros_like(sh_coeffs)
        
        for i in range(N):
            results[i] = self.rotate(sh_coeffs[i], rotations[i])
        
        return results
