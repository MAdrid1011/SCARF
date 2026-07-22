"""
LSH Hasher

Locality-Sensitive Hashing for feature signature generation.
"""
import torch
import numpy as np
from typing import Optional

from .types import FSDRConfig


def fp16_to_q24(values: torch.Tensor) -> torch.Tensor:
    """Decode normalized FP16 operands into exact signed Q1.24 integers."""
    half = values.to(torch.float16).contiguous()
    bits = half.view(torch.int16).to(torch.int64) & 0xFFFF
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if torch.any(exponent == 0x1F):
        raise ValueError("LSH operands must be finite FP16 values")
    mantissa = torch.where(exponent == 0, fraction, 0x400 + fraction)
    shift = torch.clamp(exponent - 1, min=0)
    magnitude = torch.where(exponent == 0, mantissa, mantissa << shift)
    signed = torch.where(sign == 0, magnitude, -magnitude)
    if torch.any(signed.abs() > (1 << 24)):
        raise ValueError("LSH operands must be normalized to [-1, 1]")
    return signed


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
        
        # Initialize random projection matrix [K, D]
        # Each row is a random unit vector quantized exactly as the FP16 ROM.
        generator = torch.Generator(device="cpu")
        if config.seed is not None:
            generator.manual_seed(config.seed)
        projection = torch.randn(
            self.lsh_dim, self.feature_dim, generator=generator, dtype=torch.float32
        )
        projection = projection / torch.linalg.vector_norm(
            projection, dim=1, keepdim=True
        ).clamp_min(1e-8)
        self.projection = projection.to(torch.float16)
        self.projection_q24 = fp16_to_q24(self.projection)
    
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
        if feature.ndim != 1 or feature.shape[0] != self.feature_dim:
            raise ValueError(f"feature must have shape [{self.feature_dim}]")
        return int(self.hash_batch(feature.unsqueeze(0))[0].item())
    
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
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape [N, {self.feature_dim}]")
        # Quantize normalized features and hyperplanes to the same FP16 values
        # used by the exported ROM. Float32 accumulation keeps CPU/GPU signs
        # deterministic while retaining the FP16 operand contract.
        features_f32 = features.to(torch.float32)
        features_norm = features_f32 / torch.linalg.vector_norm(
            features_f32, dim=1, keepdim=True
        ).clamp_min(1e-8)
        quantized_features = features_norm.to(torch.float16).to(torch.float32)
        feature_q24 = fp16_to_q24(quantized_features).to(
            device=features.device, dtype=torch.float64
        )
        projection_q24 = self.projection_q24.to(
            device=features.device, dtype=torch.float64
        )
        projections = feature_q24 @ projection_q24.T  # [N, K]
        
        # Pack all sign bits without per-feature device synchronization.
        bit_weights = 1 << torch.arange(
            self.lsh_dim, device=features.device, dtype=torch.int64
        )
        return ((projections >= 0).to(torch.int64) * bit_weights).sum(dim=1)
    
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
        self.projection = matrix.detach().to(device="cpu", dtype=torch.float16).clone()
        self.projection_q24 = fp16_to_q24(self.projection)


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
