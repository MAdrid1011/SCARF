"""
LSH Hasher

Locality-Sensitive Hashing for feature signature generation.
"""
import torch
import numpy as np
from typing import Optional

from .types import FSDRConfig


class LSHHasher:
    """
    Locality-Sensitive Hashing (LSH) signature generator.
    
    Uses random hyperplane hashing to map high-dimensional features
    to low-dimensional binary signatures while preserving similarity.
    
    Algorithm:
        1. Project feature vector onto K random hyperplanes
        2. Extract sign of each projection (positive → 1, negative → 0)
        3. Pack signs into K-bit integer signature
    
    Property:
        P(sign(r·x) = sign(r·y)) = 1 - arccos(cos(x,y)) / π
        
        High cosine similarity → Low Hamming distance
    
    Hardware Mapping:
        - Projection matrix stored in ROM (K × D × 16 bits)
        - K parallel dot products using MAC units
        - Sign extraction is combinational (no cycles)
        - Total: ~500 LUTs, K DSPs, 1 cycle latency
    
    Example:
        hasher = LSHHasher(config)
        signature = hasher.hash(feature)  # 16-bit integer
        
        # Batch processing
        signatures = hasher.hash_batch(features)  # [N] integers
    """
    
    def __init__(self, config: FSDRConfig):
        """
        Initialize LSH hasher with random projection matrix.
        
        Args:
            config: FSDR configuration with lsh_dim and feature_dim
        """
        self.config = config
        self.lsh_dim = config.lsh_dim
        self.feature_dim = config.feature_dim
        
        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Initialize random projection matrix [K, D]
        # Each row is a random unit vector (hyperplane normal)
        self.projection = torch.randn(self.lsh_dim, self.feature_dim)
        self.projection = self.projection / torch.norm(
            self.projection, dim=1, keepdim=True
        )
    
    def hash(self, feature: torch.Tensor) -> int:
        """
        Generate LSH signature from feature vector.
        
        Args:
            feature: [D] feature vector (will be normalized)
        
        Returns:
            K-bit integer signature (K = lsh_dim)
        
        Hardware:
            - K MAC operations (dot products)
            - K sign extractions (comparators)
            - 1 cycle latency
        """
        # Normalize input feature
        feature_norm = feature / (torch.norm(feature) + 1e-8)
        
        # Project onto random hyperplanes
        projections = self.projection @ feature_norm  # [K]
        
        # Extract signs as bits
        signature = 0
        for i in range(self.lsh_dim):
            if projections[i] >= 0:
                signature |= (1 << i)
        
        return signature
    
    def hash_batch(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate LSH signatures for a batch of features.
        
        Args:
            features: [N, D] batch of feature vectors
        
        Returns:
            [N] tensor of integer signatures
        
        Hardware:
            - Can be parallelized with N hash units
            - Or processed sequentially with 1 hash unit
        """
        N = features.shape[0]
        
        # Normalize features
        features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
        
        # Batch projection
        projections = features_norm @ self.projection.T  # [N, K]
        
        # Extract signs as bits
        signatures = torch.zeros(N, dtype=torch.int64)
        for i in range(N):
            sig = 0
            for j in range(self.lsh_dim):
                if projections[i, j] >= 0:
                    sig |= (1 << j)
            signatures[i] = sig
        
        return signatures
    
    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get the projection matrix (for serialization/hardware export).
        
        Returns:
            [K, D] projection matrix
        """
        return self.projection.clone()
    
    def set_projection_matrix(self, matrix: torch.Tensor):
        """
        Set the projection matrix (for loading from checkpoint).
        
        Args:
            matrix: [K, D] projection matrix
        """
        if matrix.shape != (self.lsh_dim, self.feature_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"expected ({self.lsh_dim}, {self.feature_dim})"
            )
        self.projection = matrix.clone()


def hamming_distance(sig1: int, sig2: int) -> int:
    """
    Compute Hamming distance between two signatures.
    
    Args:
        sig1: First signature
        sig2: Second signature
    
    Returns:
        Number of differing bits
    
    Hardware:
        - XOR operation
        - Popcount (parallel bit counting)
        - Combinational logic, no cycles
    """
    return bin(sig1 ^ sig2).count('1')


def expected_hamming_from_cosine(cosine_sim: float, lsh_dim: int = 16) -> float:
    """
    Compute expected Hamming distance from cosine similarity.
    
    Based on LSH property:
        P(sign match) = 1 - arccos(cos_sim) / π
        E[Hamming] = lsh_dim * (1 - P(sign match))
                   = lsh_dim * arccos(cos_sim) / π
    
    Args:
        cosine_sim: Cosine similarity in [-1, 1]
        lsh_dim: Number of hash bits
    
    Returns:
        Expected Hamming distance
    """
    # Clamp to valid range
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    
    # Probability of sign match
    p_match = 1.0 - np.arccos(cosine_sim) / np.pi
    
    # Expected Hamming distance
    return lsh_dim * (1.0 - p_match)
